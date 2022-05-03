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
    name="dfw",
    sources=[
        os.path.join(os.path.dirname(__file__), "csrc", "dfw.cpp"),
        os.path.join(os.path.dirname(__file__), "csrc", "dfw_kernel.cu"),
    ],
    extra_cflags=['-O0'],
    # TODO:
    # extra_cflags=['-O3'],
    is_python_module=False,
    verbose=False
)


class DFWFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                state: torch.Tensor):
        """
        Return: Output query, final hidden state
        """
        assert query.shape == key.shape
        assert query.dtype == value.dtype
        assert state.dtype == value.dtype

        outputs, final_state, old_value = torch.ops.dfw.forward(
            query, key, value,  state
        )

        ctx.save_for_backward(
            query,
            key,
            value,
            old_value,
            final_state
        )
        return outputs, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_state):
        query, key, value, old_value, final_state = ctx.saved_tensors

        assert grad_state.dtype == value.dtype
        assert grad_output.dtype == value.dtype

        res = torch.ops.dfw.backward(
            grad_output.contiguous(), grad_state.contiguous(),
            query, key, value, old_value, final_state
        )
        return tuple(res)
