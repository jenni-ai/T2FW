
import timeit

import torch
from lfw.functional import LFWFunction
from lfw.functional_dfw import DFWFunction
from lfw.torch import t2fw_torch
from tabulate import tabulate


def do_bench(fn, warmup=25, rep=100, grad_to_none=None, percentiles=[0.2, 0.8]):
    """
    Source: https://github.com/openai/triton

    Benchmark the runtime of the provided function. By default, return the median runtime of :code:`fn` along with
    the 20-th and 80-th performance percentile.

    :param fn: Function to benchmark
    :type fn: Callable
    :param warmup: Warmup time (in ms)
    :type warmup: int
    :param rep: Repetition time (in ms)
    :type rep: int
    :param grad_to_none: Reset the gradient of the provided tensor to None
    :type grad_to_none: torch.tensor, optional
    :param percentiles: Performance percentile to return in addition to the median.
    :type percentiles: list[float]
    """

    # Estimate the runtime of the function
    fn()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(5):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    estimate_ms = start_event.elapsed_time(end_event) / 5
    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2
    # doesn't contain any input data before the run
    start_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    end_event = [torch.cuda.Event(enable_timing=True) for i in range(rep)]
    cache = torch.empty(int(256e6), dtype=torch.int8, device='cuda')
    # Warm-up
    for _ in range(int(warmup / estimate_ms)):
        fn()
    # Benchmark
    for i in range(rep):
        # we don't want `fn` to accumulate gradient values
        # if it contains a backward pass. So we clear the
        # provided gradients
        if grad_to_none is not None:
            for x in grad_to_none:
                x.grad = None
        # we clear the L2 cache before each run
        cache.zero_()
        # record time of `fn`
        start_event[i].record()
        fn()
        end_event[i].record()
    torch.cuda.synchronize()
    times = torch.tensor([s.elapsed_time(e)
                         for s, e in zip(start_event, end_event)])
    percentiles = torch.quantile(times, torch.tensor(percentiles)).tolist()
    med_ms = torch.median(times).item()
    if percentiles:
        return tuple([med_ms] + percentiles)
    else:
        return med_ms


def main():
    device = 'cuda'
    providers = ('torch', 'LFWFunction', 'DFWFunction')
    seqlens = tuple(range(8, 1024, 128))
    backward = True

    dtype = torch.half
    bsz = 32
    dim = 64
    kdim = 64

    results = {}
    for provider in providers:
        print('Benchmarking:', provider)
        results[provider] = []
        for seqlen in seqlens:
            value = torch.randn(bsz, seqlen, dim, dtype=dtype, device='cuda')
            f = torch.rand_like(value)
            q, k, f_key = (torch.randn(
                bsz, seqlen, kdim, dtype=dtype, device='cuda') for _ in range(3))
            s = torch.zeros(bsz, dim, kdim, dtype=dtype, device='cuda')
            grad = torch.randn_like(value)

            input_vars = (value, f, q, k, f_key, s)
            for v in input_vars:
                v.requires_grad_()

            def wrap_bwd(fn):
                def f():
                    if backward:
                        return fn().backward(grad)
                    else:
                        return fn()
                return f

            if provider == 'torch':
                ms, *_ = do_bench(wrap_bwd(lambda: t2fw_torch(*input_vars)[0]))

            if provider == 'LFWFunction':
                ms, *_ = do_bench(
                    wrap_bwd(lambda: LFWFunction.apply(*input_vars)[0]))

            if provider == 'DFWFunction':
                ms, *_ = do_bench(
                    wrap_bwd(lambda: DFWFunction.apply(q, k, value, s)[0]))

            results[provider].append(ms)

    table = []
    for i, seqlen in enumerate(seqlens):
        table.append([seqlen] + [results[provider][i]
                     for provider in providers])
    print(tabulate(table, headers=("Seq Len",) + providers))


if __name__ == '__main__':
    main()
