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
    extra_cflags=['-O3'],
    is_python_module=False,
    verbose=False
)


class DFWFunction(torch.autograd.Function):
    """
    Implements the delta rule.
    Assumes the scalar g_t is absorbed into the key vector.
    """
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

        outputs, final_state, delta_value = torch.ops.dfw.forward(
            query, key, value,  state
        )

        ctx.save_for_backward(
            query,
            key,
            delta_value,
            final_state
        )
        return outputs, final_state

    @staticmethod
    def backward(ctx, grad_output, grad_state):
        query, key,  delta_value, final_state = ctx.saved_tensors

        assert grad_state.dtype == delta_value.dtype
        assert grad_output.dtype == delta_value.dtype

        res = torch.ops.dfw.backward(
            grad_output.contiguous(), grad_state.contiguous(),
            query, key, delta_value, final_state
        )
        return tuple(res)
