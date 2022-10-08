
import numpy as np
import torch
from t2fw.functional_dfw import DFWFunction
from t2fw.functional_lfw import LFWFunction
from t2fw.torch import t2fw_torch
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


def attention(q, k, v):
    """ Basic self-attention """
    scores = q @ k.transpose(-1, -2)
    scores = torch.softmax(scores, dim=-1)
    return scores @ v


def main():
    # variants = ('torch', 'attention', 'LFWFunction', 'DFWFunction')
    variants = ('attention', 'LFWFunction', 'DFWFunction')
    seqlens = tuple(range(0, 2048, 128))
    backward = False
    # Toggle on to simulate inference usage where state can be reused
    # and one token is generated at a time.
    test_inference = True

    dtype = torch.half
    bsz = 32
    dim = 64
    # Number of memory vectors
    kdim = 64

    results = {}
    peak_mem_results = {}
    for variant in variants:
        print('Benchmarking:', variant)
        results[variant] = []
        peak_mem_results[variant] = []
        for seqlen in seqlens:

            if variant != 'attention' and test_inference:
                seqlen = 1

            if variant == 'attention' and test_inference:
                value = torch.randn(bsz, seqlen, dim, dtype=dtype, device='cuda')
                f = torch.rand_like(value)
                k, f_key = (torch.randn(
                    bsz, seqlen, kdim, dtype=dtype, device='cuda') for _ in range(2))
                q = torch.randn(bsz, 1, kdim, dtype=dtype, device='cuda')
            else:
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

            torch.cuda.reset_peak_memory_stats()

            if variant == 'attention':
                ms, *_ = do_bench(wrap_bwd(lambda: attention(q, k, value)))

            if variant == 'torch':
                ms, *_ = do_bench(wrap_bwd(lambda: t2fw_torch(*input_vars)[0]))

            if variant == 'LFWFunction':
                ms, *_ = do_bench(
                    wrap_bwd(lambda: LFWFunction.apply(*input_vars)[0]))

            if variant == 'DFWFunction':
                ms, *_ = do_bench(
                    wrap_bwd(lambda: DFWFunction.apply(q, k, value, s)[0]))
            memory = torch.cuda.max_memory_allocated()
            # Convert to MB
            peak_mem_results[variant].append(memory * 1e-6)
            results[variant].append(ms)

    table = []
    peak_mem_table = []
    for i, seqlen in enumerate(seqlens):
        table.append([seqlen] + [results[variant][i]
                     for variant in variants])
        peak_mem_table.append([seqlen] + [peak_mem_results[variant][i]
                                          for variant in variants])

    print('Run time:')
    print(tabulate(table, headers=("Seq Len",) + variants))

    print('Peak Memory:')
    print(tabulate(peak_mem_table, headers=("Seq Len",) + variants))

    plot(variants, seqlens, results, peak_mem_results)

def plot(variants, seqlens, results, peak_mem_results):
    # Plot results
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2)
    plt.xlabel("Sequence Length (Tokens)")

    for variant in variants:
        plot_time_data = np.array(results[variant])
        plot_mem_data = np.array(peak_mem_results[variant])

        color, linestyle = colors[variant]

        axs[0].plot(
            seqlens,
            plot_time_data,
            label=variant_labels[variant],
            color=color,
            linestyle=linestyle
        )
        axs[0].set_ylabel("Execution Time (ms)")
        axs[1].plot(
            seqlens,
            plot_mem_data,
            label=variant_labels[variant],
            color=color,
            linestyle=linestyle
        )
        axs[1].set_ylabel("Peak GPU Memory (MB)")
        axs[0].get_xaxis().set_visible(False)


    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               ncol=3)
    plt.savefig('./outputs/plot.svg')
    plt.savefig('./outputs/plot.png')


variant_labels = {
    'attention': 'Attention',
    **{f'LFWFunction': f'Decay (m = {i})' for i in [64]},
    **{f'DFWFunction': f'Delta (m = {i})' for i in [64]},
}
colors = {
    'attention': ('red', '--'),
    'LFWFunction': ('#1A374D', '-'),
    'DFWFunction': ('#005502', '-.'),
    #   ('#1A374D', '-'), ('#406882', '-'), ('#6998AB', '-'),
    #   ('#005502', '-.'), ('#3A5F0B', '-.'), ('#CCFFBB', '-.')
}


if __name__ == '__main__':
    main()
