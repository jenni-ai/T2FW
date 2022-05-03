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

    # Parallel d pass. Computes d_value, s_grad_k and d_state
    d_value = []
    s_grad_ks = []

    cur_s_grad = grad_state.clone()
    state = final_state.clone()

    for t in range(value.size(1)-1, -1, -1):
        q = query[:, t]
        k = key[:, t]
        v = value[:, t]
        g = grad_output[:, t]
        old = olds[t]

        # Parallel on d (reducing m)
        d_value.append((cur_s_grad @ k.unsqueeze(-1)).squeeze(-1))

        # Move state backwards
        # Parallel on all (expansion)
        state = state - (v.unsqueeze(-1) - old) @ k.unsqueeze(-2)

        # [D, M] x [M, 1] => [D, 1] (parallel on d)
        s_grad_k = cur_s_grad @ k.unsqueeze(-1)
        s_grad_ks.append(s_grad_k)

        # Apply delta rule derivative (deriv of s_t w.r.t. s_{t-1})
        # Parallel on all (expansion)
        cur_s_grad -= s_grad_k @ k.unsqueeze(-2)
        # Apply gradient from query
        # Parallel on all (expansion)
        cur_s_grad += g.unsqueeze(-1) @ q.unsqueeze(-2)

    s_grad_ks = s_grad_ks[::-1]
    d_value = d_value[::-1]
    d_state = cur_s_grad

    # Parallel m pass. Computes d_query and d_key
    d_query = []
    d_key = []
    cur_s_grad = grad_state.clone()
    state = final_state.clone()

    for t in range(value.size(1)-1, -1, -1):
        q = query[:, t]
        k = key[:, t]
        v = value[:, t]
        g = grad_output[:, t]
        old = olds[t]
        s_grad_k = s_grad_ks[t]

        # Move state backwards
        # Parallel on all (expansion)
        state = state - (v.unsqueeze(-1) - old) @ k.unsqueeze(-2)

        # Only allow matrix-vector products (avoid matrix matrix products)
        # [1, D] x [D, M] => [1, M] (parallel on m)
        a = (v.unsqueeze(-2) @ cur_s_grad).squeeze(-2)
        # [D, M].T x [D, 1] => [M, 1] (parallel on m)
        c = (cur_s_grad.transpose(1, 2) @ old).squeeze(-1)
        # [[M, D] x [D, 1] => [M, 1] (parallel on m)
        b = (state.transpose(1, 2) @ s_grad_k).squeeze(-1)
        # Parallel on all (no reduction)
        d_key.append(a-(b+c))

        # Apply delta rule derivative (deriv of s_t w.r.t. s_{t-1})
        # Parallel on all (expansion along m)
        cur_s_grad -= s_grad_k @ k.unsqueeze(-2)
        # Apply gradient from query
        # Parallel on all (expansion)
        cur_s_grad += g.unsqueeze(-1) @ q.unsqueeze(-2)

        # Compute query gradient
        # Parallel on m (reducing d)
        d_query.append((g.unsqueeze(-2) @ state).squeeze(-2))

    d_query = d_query[::-1]
    d_key = d_key[::-1]

    return torch.stack(d_query, dim=1), torch.stack(d_key, dim=1), torch.stack(d_value, dim=1), d_state


def t2dfw_torch_bw_simple(
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

    cur_s_grad = grad_state.clone()
    state = final_state.clone()

    for t in range(value.size(1)-1, -1, -1):
        q = query[:, t]
        k = key[:, t]
        v = value[:, t]
        g = grad_output[:, t]
        old = olds[t]

        # Parallel on d (reducing m)
        d_value.append((cur_s_grad @ k.unsqueeze(-1)).squeeze(-1))

        # Move state backwards
        # Parallel on all (expansion)
        state = state - (v.unsqueeze(-1) - old) @ k.unsqueeze(-2)

        '''
        # Only allow matrix-vector products (avoid matrix matrix products)
        # [1, D] x [D, M] => [1, M] (parallel on m)
        a = (v.unsqueeze(-2) @ cur_s_grad).squeeze(-2)
        # [D, M] x [M, 1] => [D, 1], [M, D] x [D, 1] => [M, 1] (parallel on d, parallel on m)
        b = (state.transpose(1, 2) @ (cur_s_grad @ k.unsqueeze(-1))).squeeze(-1)
        # [D, M] x [M, 1] => [D, 1], [D, M].T x [D, 1] => [M, 1] (parallel on d, parallel on m)
        c = (cur_s_grad.transpose(1, 2) @ old).squeeze(-1)
        # c = (cur_s_grad.transpose(1, 2) @ (state @ k.unsqueeze(-1))).squeeze(-1)
        # Parallel on d
        d_k = a-(b+c)
        '''

        '''
        # Matrix multiplication variant
        # s_s_grad = cur_s_grad.transpose(1, 2) @ state
        # c = ((s_s_grad + s_s_grad.transpose(1, 2))
        #      @ k.unsqueeze(-1)).squeeze(-1)
        # d_k = a-c
        '''

        # [D, M] x [M, 1] => [D, 1] (parallel on d)
        s_grad_k = cur_s_grad @ k.unsqueeze(-1)

        # Only allow matrix-vector products (avoid matrix matrix products)
        # [1, D] x [D, M] => [1, M] (parallel on m)
        a = (v.unsqueeze(-2) @ cur_s_grad).squeeze(-2)
        # [D, M].T x [D, 1] => [M, 1] (parallel on m)
        c = (cur_s_grad.transpose(1, 2) @ old).squeeze(-1)
        # [[M, D] x [D, 1] => [M, 1] (parallel on m)
        b = (state.transpose(1, 2) @ s_grad_k).squeeze(-1)
        # Parallel on all (no reduction)
        d_k = a-(b+c)

        # Apply delta rule derivative (deriv of s_t w.r.t. s_{t-1})
        # Parallel on all (expansion)
        cur_s_grad -= s_grad_k @ k.unsqueeze(-2)
        # Apply gradient from query
        # Parallel on all (expansion)
        cur_s_grad += g.unsqueeze(-1) @ q.unsqueeze(-2)

        # Compute query gradient
        # Parallel on m (reducing d)
        d_query.append((g.unsqueeze(-2) @ state).squeeze(-2))
        d_key.append(d_k)

    d_query = d_query[::-1]
    d_key = d_key[::-1]
    d_value = d_value[::-1]
    d_state = cur_s_grad

    return torch.stack(d_query, dim=1), torch.stack(d_key, dim=1), torch.stack(d_value, dim=1), d_state
