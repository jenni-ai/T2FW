import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Loads the library
torch.ops.load_library("lfw.so")


class LFWFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                value: torch.Tensor,
                forget: torch.Tensor,
                query: torch.Tensor,
                key: torch.Tensor,
                f_key: torch.Tensor,
                state: torch.Tensor):
        """
        Return: Output query, final hidden state
        """
        assert value.shape == forget.shape
        assert query.shape == key.shape
        assert query.shape == f_key.shape

        assert forget.dtype == value.dtype
        assert query.dtype == value.dtype
        assert f_key.dtype == value.dtype
        assert state.dtype == value.dtype

        ckpt_states, *_ = torch.ops.lfw.forward(
            value, forget, key, f_key, state
        )

        with torch.no_grad():
            # Computing this outside the kernel is ~20% faster
            outputs = (ckpt_states @ query.unsqueeze(-1)).squeeze(-1)

            ctx.save_for_backward(
                forget,
                query,
                key,
                f_key,
                outputs,
                state,
                ckpt_states
            )
        return outputs, ckpt_states[:, -1]
