# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import ctypes

import cuda.lang as cl

from cuda.lang import _compile_host
from cuda.lang.compilation import (
    ConstantConstraint,
    KernelSignature,
    ScalarConstraint,
)


def _call(compilation, args, arg_ctypes):
    c_args = tuple(t(x) for x, t in zip(args, arg_ctypes, strict=True))
    c_arg_pointers = (ctypes.c_void_p * len(c_args))(
        *(ctypes.addressof(x) for x in c_args)
    )

    entry_type = ctypes.CFUNCTYPE(
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_void_p),
    )
    entry = entry_type(compilation.entry_address)
    status = entry(c_arg_pointers)
    ctypes.CDLL(None).fflush(None)
    return status


def test_scalar_add(capfd):
    @cl.function(host=True, tile=False)
    def add(a, b):
        x = a
        for i in range(b):
            x = x + 1
        print(x)

    compilation = _compile_host._compile(
        add,
        KernelSignature(
            (
                ScalarConstraint(cl.int32),
                ScalarConstraint(cl.int32),
            )
        ),
    )
    status = _call(compilation, (15, 27), (ctypes.c_int32, ctypes.c_int32))
    assert status == 0
    assert capfd.readouterr().out == "42\n"


def test_scalar_add_constant(capfd):
    @cl.function(host=True, tile=False)
    def add_const(a, x: cl.Constant[int]):
        print(a + x)

    compilation = _compile_host._compile(
        add_const,
        KernelSignature(
            (
                ScalarConstraint(cl.int32),
                ConstantConstraint(42),
            )
        ),
    )
    status = _call(compilation, (0,), (ctypes.c_int32,))
    assert status == 0
    assert capfd.readouterr().out == "42\n"
