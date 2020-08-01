from typing import Optional, Union

import torch
import torch.nn as nn


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            self.add_module("prototypes" + str(i),
                            nn.Linear(output_dim, k, bias=False))

    def forward(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, "prototypes" + str(i))(x))
        return out


class SwAV(nn.Module):
    def __init__(
            self,
            base_model: nn.Module,
            base_model_dim: int,
            normalize=False,
            output_dim=0,
            hidden_mlp=0,
            nmb_prototypes=0,
            eval_mode=False,
    ):
        super().__init__()
        self.base_model = base_model
        self.base_model_dim = base_model_dim

        # normalize output features
        self.l2norm = normalize

        # projection head
        if output_dim == 0:
            self.projection_head: Optional[nn.Module] = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(base_model_dim, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(base_model_dim, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        self.prototypes: Optional[Union[MultiPrototypes, nn.Linear]] = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

    def forward_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = nn.functional.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            _out = self.base_model(
                torch.cat(inputs[start_idx: end_idx]).cuda(non_blocking=True)
            )
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        return self.forward_head(output)
