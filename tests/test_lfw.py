import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from lfw.torch import t2fw_torch
from lfw.functional import LFWFunction


def uniform_gate_init(tensor):
    # Uniform Gate Initialization
    # https://arxiv.org/pdf/1910.09890.pdf
    bias_size = tensor.size(-1)
    u = torch.rand(bias_size, dtype=tensor.dtype,
                   device=tensor.device) * (1-1/bias_size*2) + (1/bias_size)
    return tensor.zero_().add_(-torch.log(1/u-1))


def bld_gen():
    for bsz in range(1, 16, 4):
        for dim in range(1, 64, 32):
            for seqlen in range(1, 512, 128):
                yield bsz, seqlen, dim
    yield 2, 1024, 32


class Test(unittest.TestCase):
    def test_cpp(self):
        torch.manual_seed(1)
        dtype = torch.half

        for bsz, seqlen, dim in bld_gen():
            # bsz, seqlen, dim = (1, 256, 32)
            kdim = math.ceil(dim / 2)
            print('test_cpp', bsz, seqlen, dim, kdim)

            x = torch.randn(bsz, seqlen, dim, dtype=dtype, device='cuda')
            f = torch.sigmoid(uniform_gate_init(torch.empty_like(x)))
            q, k, f_key = (torch.randn(
                bsz, seqlen, kdim, dtype=dtype, device='cuda') for _ in range(3))
            f_key = torch.sigmoid(uniform_gate_init(f_key))
            s = torch.randn(bsz, dim, kdim, dtype=dtype, device='cuda')

            input_vars = (x, f, q, k, f_key, s)
            for v in input_vars:
                v.requires_grad_()

            ref_output, ref_state = t2fw_torch(*input_vars)
            grad = torch.randn_like(ref_output)
            grad_state = torch.randn_like(ref_state)
            ref_output.backward(grad, retain_graph=True)
            ref_state.backward(grad_state)
            ref_grads = tuple(v.grad for v in input_vars)

            # Clear grad
            for v in input_vars:
                v.grad = None

            fast_output, fast_state = LFWFunction.apply(*input_vars)
            fast_output.backward(grad, retain_graph=True)
            fast_state.backward(grad_state)
            fast_grads = tuple(v.grad for v in input_vars)

            try:
                assert torch.allclose(ref_output, fast_output, atol=1e-2, rtol=1e-1), (
                    f'The maximum difference between ref and fast is '
                    f'{torch.max(torch.abs(ref_output - fast_output))}'
                )
            except Exception as e:
                print('ref_output', ref_output, ref_output.size())
                print('fast_output', fast_output, fast_output.size())
                raise e

            assert len(ref_grads) == len(fast_grads)

            errored = None

            try:
                assert torch.allclose(ref_output, fast_output, atol=1e-2, rtol=1), (
                    f'The maximum difference between ref and fast is '
                    f'{torch.max(torch.abs(ref_output - fast_output))}'
                )
            except Exception as e:
                print('ref_output', ref_output)
                print('fast_output', fast_output)
                print()
                errored = e

            for i, (ref, out) in enumerate(zip(ref_grads, fast_grads)):
                try:
                    # Estimation of 0.3 should lead to convergence
                    assert torch.allclose(ref, out, atol=0.1, rtol=1e-1), (
                        f'The maximum difference between ref and out is '
                        f'{torch.max(torch.abs(ref - out))}'
                    )
                except Exception as e:
                    print(f'ref grad {i} {ref}')
                    print(f'out grad {i} {out}')
                    print()
                    errored = e

            if errored:
                raise errored


if __name__ == '__main__':
    unittest.main()
