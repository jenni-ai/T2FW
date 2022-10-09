# Fine-Tuning Pre-trained Transformers into Decaying Fast Weights
CUDA Kernels for the paper "Fine-Tuning Pre-trained Transformers into Decaying Fast Weights" in EMNLP 2022.

## Abstract
Autoregressive Transformers are strong language models but incur O(T) complexity during per-token generation due to the self-attention mechanism. Recent work proposes kernel-based methods to approximate causal self-attention by replacing it with recurrent formulations with various update rules and feature maps to achieve O(1) time and memory complexity. We explore these approaches and find that they are unnecessarily complex, and propose a simple alternative - decaying fast weights - that runs fast on GPU, outperforms prior methods, and retains 99\% of attention's performance for GPT-2.
We also show competitive performance on WikiText-103 against more complex attention substitutes.

## Install
Installing the package by cloning the repository, then run:
```
python setup.py install
```

## Usage
The provided function only works on CUDA.
JIT may take some time to compile.
Recommended to use `torch.half` datatype.

```py3
import torch
from t2fw.functional_lfw import LFWFunction
bsz = 1
seqlen = 4
dim = 4
kdim = 4

x = torch.randn(bsz, seqlen, dim, dtype=torch.half, device='cuda')
f = torch.empty_like(x)
q, k, f_key = (torch.randn(bsz, seqlen, kdim, dtype=torch.half, device='cuda') for _ in range(3))
s = torch.randn(bsz, dim, kdim, dtype=torch.half, device='cuda')

outputs, next_state = LFWFunction.apply(x, f, q, k, f_key, s)
```

## Unit Tests
Check correctness of forward and backward pass against Torchscript implementation.
```
python -m unittest tests/*.py
```

Benchmark running time against Torchscript implementation.
```
python -m tests.benchmark
```