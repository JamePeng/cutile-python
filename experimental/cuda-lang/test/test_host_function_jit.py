# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes
from typing import Annotated
from unittest.mock import patch

import pytest
import torch

import cuda.lang as cl
from cuda.lang._compile_host import _compile as compile_host
from cuda.lang._exception import TypeCheckingError


def test_host_jit_no_specialization(capfd):
    @cl.host_entry
    def add(a, b):
        print(a + b)

    with patch(
        "cuda.lang._compile_host._compile",
        wraps=compile_host,
    ) as compile_host_mock:
        add(15, 27)
        add(16, 26)

    ctypes.CDLL(None).fflush(None)

    assert compile_host_mock.call_count == 1
    assert capfd.readouterr().out == "42\n42\n"


def test_host_jit_specializes_argument_structure(capfd):
    @cl.host_entry
    def report(value):
        print(42)

    with patch(
        "cuda.lang._compile_host._compile",
        wraps=compile_host,
    ) as compile_host_mock:
        report(0)
        report((0,))
        report(1)
        report((1,))

    ctypes.CDLL(None).fflush(None)

    assert compile_host_mock.call_count == 2
    assert capfd.readouterr().out == "42\n42\n42\n42\n"


def test_host_jit_specialize_static_shape():
    @cl.kernel
    def write_static_shape(x, out):
        if cl.thread_index(0) == 0:
            out[0] = x.shape[0]

    @cl.host_entry
    def entry(
        stream: cl.ScalarInt64,
        x: Annotated[
            cl.Array,
            cl.ArrayAnnotation(static_shape_dims=(0,)),
        ],
        out,
    ):
        cl.launch(stream, (1,), (1,), write_static_shape, (x, out))

    stream = torch.cuda.current_stream().cuda_stream
    output = torch.zeros(1, dtype=torch.int32, device="cuda")
    with patch(
        "cuda.lang._compile_host._compile",
        wraps=compile_host,
    ) as compile_host_mock:
        entry(stream, torch.zeros(4, dtype=torch.int32, device="cuda"), output)
        assert compile_host_mock.call_count == 1
        assert output.item() == 4

        entry(stream, torch.zeros(8, dtype=torch.int32, device="cuda"), output)
        assert compile_host_mock.call_count == 2
        assert output.item() == 8


def test_host_jit_two_kernel_specializations():
    @cl.kernel
    def write(value: cl.Constant[int], output):
        output[0] = value

    @cl.host_entry
    def entry(stream: cl.ScalarInt64, first, second):
        cl.launch(stream, (1,), (1,), write, (3, first))
        cl.launch(stream, (1,), (1,), write, (7, second))

    stream = torch.cuda.current_stream().cuda_stream
    first = torch.zeros(1, dtype=torch.int32, device="cuda")
    second = torch.zeros(1, dtype=torch.int32, device="cuda")
    with (
        patch(
            "cuda.lang._compile_host._compile",
            wraps=compile_host,
        ) as compile_host_mock,
        patch.object(write, "_compile", wraps=write._compile) as compile_kernel_mock,
    ):
        entry(stream, first, second)
        entry(stream, first, second)

    torch.cuda.synchronize()
    assert (first.item(), second.item()) == (3, 7)
    assert compile_host_mock.call_count == 1
    assert compile_kernel_mock.call_count == 2


def test_host_jit_rejects_call_to_host_entry():
    @cl.host_entry
    def inner():
        pass

    @cl.host_entry
    def outer():
        inner()

    with pytest.raises(
        TypeCheckingError,
        match="Cannot call an object of type cuda.lang host entry",
    ):
        outer()
