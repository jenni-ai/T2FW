# Fine-Tuning Pre-trained Transformers into Decaying Fast Weights


## Install
```
python setup.py install
```

## Usage
The provided function only works on CUDA.
JIT may take some time to compile.
Recommended to use `torch.half` datatype.

```py3
import torch
from lfw.functional import LFWFunction
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
```
python -m unittest tests/*.py
```