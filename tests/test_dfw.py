import math
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from t2fw.torch import t2dfw_torch, t2dfw_torch_bw
from t2fw.functional_dfw import DFWFunction


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
    yield 2, 1024, 96


class Test(unittest.TestCase):
    def test_cpp(self):
        torch.manual_seed(1)
        # Pure additive may cause numerical issue under fp16
        dtype = torch.float

        for bsz, seqlen, dim in bld_gen():
            mdim = math.ceil(dim / 2)
            # bsz, seqlen, dim, mdim = (1, 4, 4, 4)
            print('test_cpp', bsz, seqlen, dim, mdim)

            value = torch.randn(bsz, seqlen, dim, dtype=dtype, device='cuda')
            query, key = (
                torch.randn(bsz, seqlen, mdim, dtype=dtype, device='cuda') for _ in range(2)
            )
            # Allow keys to be scaled
            key = torch.softmax(key, dim=-1) * \
                torch.sigmoid(key.mean(dim=-1, keepdim=True))

            state = torch.randn(bsz, dim, mdim, dtype=dtype,
                                device='cuda')

            input_vars = (query, key, value, state)
            for var in input_vars:
                var.requires_grad_()

            ref_output, ref_state = t2dfw_torch(*input_vars)
            grad = torch.randn_like(ref_output)
            grad_state = torch.randn_like(ref_state)
            ref_output.backward(grad, retain_graph=True)
            ref_state.backward(grad_state)
            ref_grads = tuple(v.grad for v in input_vars)

            # Clear grad
            for var in input_vars:
                var.grad = None

            fast_output, fast_state = DFWFunction.apply(*input_vars)
            fast_output.backward(grad, retain_graph=True)
            fast_state.backward(grad_state)
            fast_grads = tuple(v.grad for v in input_vars)
            # with torch.no_grad():
            #     fast_grads = tuple(t2dfw_torch_bw(
            #         grad, grad_state,
            #         query, key, value, state, fast_state
            #     ))

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
