# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any

import cuda.lang as cl
import torch
import pytest
import cuda.core

from cuda.tile._cext import cconv_v3_enabled


def test_too_many_kwargs():
    @cl.kernel()
    def kernel():
        pass

    bad_kwargs = {f"kw{i}": i for i in range(20)}

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (), **bad_kwargs)


def test_cuda_core_buffer_kernel_arg():

    @cl.kernel
    def kernel(out):
        i = cl.thread_index(0)
        out[i] = i + 1

    device = cuda.core.Device(0)
    device.set_current()
    stream = device.create_stream()
    buffer = device.allocate(32, stream=stream)

    cl.launch(int(stream.handle), (1,), (32,), kernel, (buffer,))
    stream.sync()

    out = torch.from_dlpack(buffer)
    assert out.tolist() == list(range(1, 33))


def test_cuda_core_buffer_torch_view_kernel_arg():

    @cl.kernel
    def kernel(out):
        i = cl.thread_index(0)
        out[i] = i + 0.5

    device = cuda.core.Device(0)
    device.set_current()
    stream = device.create_stream()
    buffer = device.allocate(32 * 4, stream=stream)
    stream.sync()
    out = torch.from_dlpack(buffer).view(torch.float32)

    cl.launch(int(stream.handle), (1,), (32,), kernel, (out,))
    stream.sync()

    expected = [i + 0.5 for i in range(32)]
    assert out.tolist() == expected
    assert torch.from_dlpack(buffer).view(torch.float32).tolist() == expected


@pytest.mark.skipif(not cconv_v3_enabled(), reason="Requires cconv3 enabled")
def test_dataclass_kernel_arg():
    @dataclass(frozen=True)
    class KernelArgs:
        out: Any
        scale: int
        bias: float

    @cl.kernel()
    def kernel(a):
        i = cl.thread_index(0)
        a.out[i] = i * a.scale + a.bias

    x = torch.zeros((4,), dtype=torch.float32, device="cuda:0")
    cl.launch(torch.cuda.current_stream(), (1,), (4,), kernel, (KernelArgs(x, 10, 0.5),))
    assert x.tolist() == [0.5, 10.5, 20.5, 30.5]
