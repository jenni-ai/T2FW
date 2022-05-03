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


# @torch.jit.script
def t2dfw_torch_bw(
    grad_output: torch.Tensor,
    grad_state: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    state: torch.Tensor,
    final_state: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Equivalent Torchscript implementation of backward pass for reference """
    # Recompute old values
    olds = []
    for t in range(value.size(1)):
        old = state @ key[:, t].unsqueeze(-1)
        new = value[:, t].unsqueeze(-1)
        out = (state @ query[:, t].unsqueeze(-1)).squeeze(-1)

        # [B, D, K]
        state = state + (new - old) @ key[:, t].unsqueeze(-2)
        olds.append(old)

    # Compute gradients
    d_query = []
    d_key = []
    d_value = []

    cur_s_grad = grad_state
    state = final_state

    for t in range(value.size(1)-1, -1, -1):
        q = query[:, t]
        k = key[:, t]
        v = value[:, t]
        g = grad_output[:, t]
        old = olds[t]

        d_value.append((cur_s_grad @ k.unsqueeze(-1)).squeeze(-1))

        cur_s_grad2 = cur_s_grad.clone()

        # Apply delta rule derivative (deriv of s_t w.r.t. s_{t-1})
        cur_s_grad -= cur_s_grad @ k.unsqueeze(-1) @ k.unsqueeze(-2)
        # Apply gradient from query
        cur_s_grad += g.unsqueeze(-1) @ q.unsqueeze(-2)

        # Move state backwards
        state = state - (v.unsqueeze(-1) - old) @ k.unsqueeze(-2)

        a = (v.unsqueeze(-2) @ cur_s_grad2).squeeze(-2)
        b = -(state.transpose(1, 2) @ (cur_s_grad2 @ k.unsqueeze(-1))).squeeze(-1)
        c = -(k.unsqueeze(-2) @ state.transpose(1, 2) @ cur_s_grad2).squeeze(-2)
        d_k = a+b+c

        d_query.append((g.unsqueeze(-2) @ state).squeeze(-2))
        d_key.append(d_k)

    d_query = d_query[::-1]
    d_key = d_key[::-1]
    d_value = d_value[::-1]
    d_state = cur_s_grad

    return torch.stack(d_query, dim=1), torch.stack(d_key, dim=1), torch.stack(d_value, dim=1), d_state
