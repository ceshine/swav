# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import argparse
import math
import os
import shutil
import time
from logging import getLogger
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import apex

from src.utils import (
    bool_flag,
    initialize_exp,
    restart_from_checkpoint,
    fix_random_seeds,
    AverageMeter
)
from src.multicropdataset import MultiCropDatasetAlt
from src.bit_models import KNOWN_MODELS as BIT_MODELS
from src.model_wrapper import SwAV

logger = getLogger()

parser = argparse.ArgumentParser(description="Implementation of SwAV")

#########################
#### data parameters ####
#########################
parser.add_argument("--data_path", type=str, default="/path/to/imagenet/",
                    help="path to dataset repository")
parser.add_argument("--nmb_crops", type=int, default=[2], nargs="+",
                    help="list of number of crops (example: [2, 6])")
parser.add_argument("--size_crops", type=int, default=[224], nargs="+",
                    help="crops resolutions (example: [224, 96])")
parser.add_argument("--min_scale_crops", type=float, default=[0.14], nargs="+",
                    help="argument in RandomResizedCrop (example: [0.14, 0.05])")
parser.add_argument("--max_scale_crops", type=float, default=[1], nargs="+",
                    help="argument in RandomResizedCrop (example: [1., 0.14])")

#########################
## swav specific params #
#########################
parser.add_argument("--crops_for_assign", type=int, nargs="+", default=[0, 1],
                    help="list of crops id used for computing assignments")
parser.add_argument("--temperature", default=0.1, type=float,
                    help="temperature parameter in training loss")
parser.add_argument("--epsilon", default=0.05, type=float,
                    help="regularization parameter for Sinkhorn-Knopp algorithm")
parser.add_argument("--sinkhorn_iterations", default=3, type=int,
                    help="number of iterations in Sinkhorn-Knopp algorithm")
parser.add_argument("--feat_dim", default=128, type=int,
                    help="feature dimension")
parser.add_argument("--nmb_prototypes", default=3000, type=int,
                    help="number of prototypes")
parser.add_argument("--queue_length", type=int, default=0,
                    help="length of the queue (0 for no queue)")
# parser.add_argument("--epoch_queue_starts", type=int, default=0,
#                     help="from this epoch, we start using a queue")

#########################
#### optim parameters ###
#########################
parser.add_argument("--epochs", default=100, type=int,
                    help="number of total epochs to run")
parser.add_argument("--batch_size", default=64, type=int,
                    help="batch size per gpu, i.e. how many unique instances per gpu")
parser.add_argument("--base_lr", default=4.8, type=float,
                    help="base learning rate")
parser.add_argument("--final_lr", type=float, default=0,
                    help="final learning rate")
parser.add_argument("--freeze_prototypes_niters", default=313, type=int,
                    help="freeze the prototypes during this many iterations from the start")
parser.add_argument("--wd", default=1e-6, type=float, help="weight decay")
parser.add_argument("--warmup_epochs", default=.5,
                    type=float, help="number of warmup epochs")
parser.add_argument("--start_warmup", default=0, type=float,
                    help="initial warmup learning rate")

#########################
#### other parameters ###
#########################
parser.add_argument("--arch", default="BiT-M-R50x1",
                    type=str, help="convnet architecture")
parser.add_argument("--pretrained_path", default="",
                    type=str, help="folder where the pretrained weights are stored")
parser.add_argument("--hidden_mlp", default=2048, type=int,
                    help="hidden layer dimension in projection head")
parser.add_argument("--workers", default=4, type=int,
                    help="number of data loading workers")
parser.add_argument("--checkpoint_freq", type=int, default=2000,
                    help="Save the model periodically")
parser.add_argument("--use_fp16", type=bool_flag, default=True,
                    help="whether to train with mixed precision or not")
# parser.add_argument("--sync_bn", type=str,
#                     default="pytorch", help="synchronize bn")
parser.add_argument("--dump_path", type=str, default=".",
                    help="experiment dump path for checkpoints and log")
parser.add_argument("--seed", type=int, default=31, help="seed")


def main():
    global args
    args = parser.parse_args()
    args.rank = 0
    fix_random_seeds(args.seed)
    logger, training_stats = initialize_exp(args, "epoch", "loss")

    # build data
    train_dataset = MultiCropDatasetAlt(
        args.data_path,
        args.size_crops,
        args.nmb_crops,
        args.min_scale_crops,
        args.max_scale_crops,
    )
    # sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        # sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info("Building data done with {} images loaded.".format(
        len(train_dataset)))

    # build model
    base_model = BIT_MODELS[args.arch](head_size=-1)
    if args.pretrained_path:
        logger.info("Loading pretrained model")
        base_model.load_from(np.load(f"{args.pretrained_path}/{args.arch}.npz"))
    model = SwAV(
        base_model,
        base_model_dim=base_model.width_factor * 2048,
        normalize=True,
        hidden_mlp=args.hidden_mlp,
        output_dim=args.feat_dim,
        nmb_prototypes=args.nmb_prototypes
    )
    # copy model to GPU
    model = model.cuda()
    logger.info(model)
    logger.info("Building model done.")

    # build optimizer
    # optimizer = torch.optim.SGD(
    #     model.parameters(),
    #     lr=args.base_lr,
    #     momentum=0.9,
    #     weight_decay=args.wd,
    # )
    # optimizer = LARC(optimizer=optimizer, trust_coefficient=0.001, clip=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.wd
    )
    warmup_lr_schedule = np.linspace(
        args.start_warmup, args.base_lr, int(len(train_loader) * args.warmup_epochs))
    iters = np.arange(int(len(train_loader) * (args.epochs - args.warmup_epochs)))
    cosine_lr_schedule = np.array(
        [
            args.final_lr +
            0.5 * (args.base_lr - args.final_lr) *
            (1 + math.cos(math.pi * t / (len(train_loader) * (args.epochs - args.warmup_epochs))))
            for t in iters
        ]
    )
    lr_schedule = np.concatenate((warmup_lr_schedule, cosine_lr_schedule))
    logger.info("Building optimizer done.")

    # init mixed precision
    if args.use_fp16:
        model, optimizer = apex.amp.initialize(
            model, optimizer, opt_level="O2")
        logger.info("Initializing mixed precision done.")

    # optionally resume from a checkpoint
    # TODO: set another seed for data loading when restoring
    to_restore = {"step": 0}
    restart_from_checkpoint(
        os.path.join(args.dump_path, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=model,
        optimizer=optimizer,
        amp=apex.amp,
    )
    start_step = to_restore["step"]

    # build the queue
    queue = None
    queue_path = os.path.join(
        args.dump_path, "queue.pth")
    if os.path.isfile(queue_path):
        queue = torch.load(queue_path)["queue"]
    # the queue needs to be divisible by the batch size
    args.queue_length -= args.queue_length % (args.batch_size)

    cudnn.benchmark = True

    # train the network
    scores, queue = train(
        train_loader, model,
        optimizer, start_step, lr_schedule, queue_path, args
    )
    # for epoch in range(start_epoch, args.epochs):
    # train the network for one epoch
    # logger.info("============ Starting epoch %i ... ============" % epoch)

    # # optionally starts a queue
    # if args.queue_length > 0 and epoch >= args.epoch_queue_starts and queue is None:
    #     queue = torch.zeros(
    #         len(args.crops_for_assign),
    #         args.queue_length,
    #         args.feat_dim,
    #     ).cuda()

    # train the network
    # scores, queue = train(train_loader, model,
    #                       optimizer, start_iter, lr_schedule, queue, args)
    # training_stats.update(scores)

    # save checkpoints
    # save_dict = {
    #     "epoch": epoch + 1,
    #     "state_dict": model.state_dict(),
    #     "optimizer": optimizer.state_dict(),
    # }
    # if args.use_fp16:
    #     save_dict["amp"] = apex.amp.state_dict()
    # torch.save(
    #     save_dict,
    #     os.path.join(args.dump_path, "checkpoint.pth.tar"),
    # )
    # if epoch % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
    #     shutil.copyfile(
    #         os.path.join(args.dump_path, "checkpoint.pth.tar"),
    #         os.path.join(args.dump_checkpoints,
    #                      "ckp-" + str(epoch) + ".pth"),
    #     )
    # if queue is not None:
    #     torch.save({"queue": queue}, queue_path)


def train(train_loader, model, optimizer, start_step, lr_schedule, queue_path, args):
    queue: Optional[torch.Tensor] = None
    if args.queue_length > 0:
        queue = torch.zeros(
            len(args.crops_for_assign),
            args.queue_length,
            args.feat_dim,
        ).cuda()
        if args.use_fp16:
            queue = queue.half()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    softmax = nn.Softmax(dim=1).cuda()
    model.train()
    use_the_queue = False
    step = start_step

    while step < args.epochs * len(train_loader):
        end = time.time()
        for inputs in train_loader:
            # measure data loading time
            data_time.update(time.time() - end)

            # update learning rate
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[step]

            # normalize the prototypes
            with torch.no_grad():
                w = model.prototypes.weight.data.clone()
                w = nn.functional.normalize(w, dim=1, p=2)
                model.prototypes.weight.copy_(w)

            # ============ multi-res forward passes ... ============
            embedding, output = model(inputs)
            embedding = embedding.detach()
            bs = inputs[0].size(0)

            # ============ swav loss ... ============
            loss = 0
            for i, crop_id in enumerate(args.crops_for_assign):
                with torch.no_grad():
                    out = output[bs * crop_id: bs * (crop_id + 1)]

                    # time to use the queue
                    if queue is not None:
                        if use_the_queue or not torch.all(queue[i, -1, :] == 0):
                            use_the_queue = True
                            out = torch.cat((torch.mm(
                                queue[i],
                                model.prototypes.weight.t()
                            ).float(), out))
                        # fill the queue
                        queue[i, bs:] = queue[i, :-bs].clone()
                        queue[i, :bs] = embedding[crop_id * bs: (crop_id + 1) * bs]
                    # get assignments
                    q = torch.exp(out / args.epsilon).t()
                    q = sinkhorn(q, args.sinkhorn_iterations)[-bs:]

                # cluster assignment prediction
                subloss = 0
                for v in np.delete(np.arange(np.sum(args.nmb_crops)), crop_id):
                    p = softmax(output[bs * v: bs * (v + 1)] / args.temperature)
                    subloss -= torch.mean(torch.sum(q * torch.log(p), dim=1))
                loss += subloss / (np.sum(args.nmb_crops) - 1)
            loss /= len(args.crops_for_assign)

            # ============ backward and optim step ... ============
            optimizer.zero_grad()
            if args.use_fp16:
                with apex.amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # cancel some gradients
            if step < args.freeze_prototypes_niters:
                for name, p in model.named_parameters():
                    if "prototypes" in name:
                        p.grad = None
            optimizer.step()

            # ============ misc ... ============
            losses.update(loss.item(), inputs[0].size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            if (step + 1) % 500 == 0:
                logger.info(
                    "Epoch: [{0}][{1}]\t"
                    "Time {batch_time.val:.6f} ({batch_time.avg:.6f})\t"
                    "Data {data_time.val:.6f} ({data_time.avg:.6f})\t"
                    "Loss {loss.val:.6f} ({loss.avg:.6f})\t"
                    "Lr: {lr:.8f}".format(
                        step // len(train_loader),
                        step % len(train_loader) + 1,
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        lr=optimizer.param_groups[0]["lr"],
                    )
                )
            if (step + 1) % args.checkpoint_freq == 0:
                save_dict = {
                    "step": step + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                if args.use_fp16:
                    save_dict["amp"] = apex.amp.state_dict()
                torch.save(
                    save_dict,
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                )
                shutil.copyfile(
                    os.path.join(args.dump_path, "checkpoint.pth.tar"),
                    os.path.join(args.dump_checkpoints,
                                 "ckp-" + str(step + 1) + ".pth"),
                )
                if queue is not None:
                    torch.save({"queue": queue}, queue_path)
            step += 1
    return losses.avg, queue


def sinkhorn(Q, nmb_iters):
    with torch.no_grad():
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        u = torch.zeros(Q.shape[0]).cuda(non_blocking=True)
        r = torch.ones(Q.shape[0]).cuda(non_blocking=True) / Q.shape[0]
        c = torch.ones(Q.shape[1]).cuda(
            non_blocking=True) / Q.shape[1]

        curr_sum = torch.sum(Q, dim=1)

        for it in range(nmb_iters):
            u = curr_sum
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            curr_sum = torch.sum(Q, dim=1)
        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()


if __name__ == "__main__":
    main()
