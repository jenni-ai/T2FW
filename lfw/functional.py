import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# Loads the library
from pathlib import Path

path = Path(os.path.dirname(os.path.abspath(__file__)))
torch.ops.load_library(os.path.join(
    path.parent, 'build', 'lib.linux-x86_64-3.8', 'lfw.so'))


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

        outputs, ckpt_states = torch.ops.lfw.forward(
            value, forget, query, key, f_key, state
        )

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
