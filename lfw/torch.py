import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@torch.jit.script
def t2fw_torch(
        x: torch.Tensor,
        f: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        f_key: torch.Tensor,
        state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Equivalent Torchscript implementation for reference """

    outputs = torch.jit.annotate(List[torch.Tensor], [])

    for t in range(x.size(1)):
        f_full = f[:, t].unsqueeze(-1) @ f_key[:, t].unsqueeze(-2)
        x_full = x[:, t].unsqueeze(-1) @ key[:, t].unsqueeze(-2)
        # [B, D, K]
        state = state * f_full + x_full

        out = (state @ query[:, t].unsqueeze(-1)).squeeze(-1)
        outputs.append(out)
    outputs = torch.stack(outputs, dim=1)

    return outputs, state


@torch.jit.script
def t2dfw_torch(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Equivalent Torchscript implementation for reference """

    outputs = torch.jit.annotate(List[torch.Tensor], [])

    for t in range(value.size(1)):
        old = state @ key[:, t].unsqueeze(-1)
        new = value[:, t].unsqueeze(-1)
        out = (state @ query[:, t].unsqueeze(-1)).squeeze(-1)

        # [B, D, K]
        state = state + (new - old) @ key[:, t].unsqueeze(-2)

        outputs.append(out)
    outputs = torch.stack(outputs, dim=1)

    return outputs, state
