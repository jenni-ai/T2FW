import math
import os
# Loads the library
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

load(
    name="lfw",
    sources=[
        os.path.join(os.path.dirname(__file__), "csrc", "lfw.cpp"),
        os.path.join(os.path.dirname(__file__), "csrc", "lfw_kernel.cu"),
    ],
    extra_cflags=['-O3'],
    is_python_module=False,
    verbose=False
)


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
            value,
            forget,
            query,
            key,
            f_key,
            outputs,
            state,
            ckpt_states
        )
        return outputs, ckpt_states[:, -1]

    @staticmethod
    def backward(ctx, grad_output, grad_state):
        value, forget, query, key, f_key, outputs, state, ckpt_states = ctx.saved_tensors

        assert grad_state.dtype == value.dtype
        assert grad_output.dtype == value.dtype

        res = torch.ops.lfw.backward(
            grad_output, grad_state,
            value, forget, query, key, f_key, outputs, state, ckpt_states
        )
        return tuple(res)
