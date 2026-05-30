<!--- SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

# cuda.lang

`cuda.lang` compiles Python to cubin, providing a SIMT programming model.

```python
import cuda.lang as cl
import torch

@cl.kernel
def saxpy(
    N: cl.Constant[int],
    a: cl.Constant[float],
    X, Y
):
    idx = cl.thread_idx(0) + cl.block_idx(0) * cl.block_dim(0)
    if idx < N:
        Y[idx] = a * X[idx] + Y[idx]

N = 256
alpha = 2.0
X = torch.ones(N, dtype=torch.float32, device="cuda")
Y = torch.ones(N, dtype=torch.float32, device="cuda")
expected = (alpha * X + Y).cpu()
cl.launch(
  torch.cuda.current_stream(),
  (64,),
  (64,),
  saxpy,
  (N, alpha, X, Y),
)
assert torch.allclose(expected, Y.cpu())
```

See `test/examples` for more example programs.
