# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
from cuda.lang._ir import ir
from cuda.lang.compilation import KernelSignature


def test_compile_simt_no_ptx():
    def kernel():
        pass

    cres = cl.compile_simt(kernel, [KernelSignature([])])
    assert isinstance(cres.final_ir, ir.Region)
    assert isinstance(cres.mlir, str)
    assert isinstance(cres.cubin, bytes)
    assert isinstance(cres.ptx, type(None))


def test_compile_simt_ptx(log_ptx):
    def kernel():
        pass

    cres = cl.compile_simt(kernel, [KernelSignature([])])
    assert isinstance(cres.final_ir, ir.Region)
    assert isinstance(cres.mlir, str)
    assert isinstance(cres.cubin, bytes)
    assert isinstance(cres.ptx, str)
    sym = cres.kernel_signatures[0].symbol
    assert f'.visible .entry {sym}()' in cres.ptx
