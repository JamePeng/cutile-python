# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import torch
from torch.testing import assert_close

from cuda.lang._compile import get_compute_capability
from cuda.lang._compilers import get_nvvm, PtxCompiler
from cuda.lang._llvm_bitcode import BitcodeBuilder, DATALAYOUT_PTX, CallingConvention, Binop
from cuda.tile._cext import TileDispatcher

from cuda.tile._annotated_function import LeafAnnotationNode
import cuda.lang as cl


def _make_a_plus_b_bitcode() -> bytes:
    builder = BitcodeBuilder(target_triple="nvptx64-nvidia-cuda", data_layout=DATALAYOUT_PTX)
    tt = builder.type_table

    builder.append_nvvm_version_metadata(2, 0)

    with builder.function("llvm.nvvm.read.ptx.sreg.tid.x",
                          tt.function(tt.I32, ())) as f:
        tid_x = f.value

    with builder.function("aplusb",
                          tt.function(tt.VOID,
                                      (tt.P0, tt.I32, tt.I32,
                                       tt.P0, tt.I32, tt.I32,
                                       tt.P0, tt.I32, tt.I32)),
                          calling_convention=CallingConvention.PTX_Kernel) as foobar:
        aptr, asize, astride, bptr, bsize, bstride, cptr, csize, cstride = foobar.parameters
        tid = builder.call(tt.function(tt.I32, ()), tid_x, ())
        ap = builder.get_element_ptr(tt.F32, aptr, tid)
        a = builder.load(tt.F32, ap, alignment=4)
        bp = builder.get_element_ptr(tt.F32, bptr, tid)
        b = builder.load(tt.F32, bp, alignment=4)
        cp = builder.get_element_ptr(tt.F32, cptr, tid)
        c = builder.binop(tt.I32, Binop.ADD, a, b)
        builder.store(cp, c, alignment=4)
        builder.ret()
    return builder.build()


def test_a_plus_b():
    bitcode = _make_a_plus_b_bitcode()
    cc = get_compute_capability()

    # Compile bitcode to PTX using NVVM
    nvvm = get_nvvm()
    program = nvvm.create_program()
    program.add_module(bitcode, "main")
    ptx = program.compile(["-arch=" + cc.arch])

    ptx_compiler: PtxCompiler = PtxCompiler.get()
    cubin = ptx_compiler.compile(ptx, cc.gpu_name)

    kernel = _HackKernel(cubin, "aplusb", 3)
    a = torch.ones(32, dtype=torch.float32, device="cuda")
    b = torch.arange(32, dtype=torch.float32, device="cuda")
    ref = a + b
    c = torch.zeros_like(a)
    cl.launch(torch.cuda.current_stream(), (1,), (32,), kernel, (a, b, c))
    assert_close(c, ref, rtol=0, atol=0)


class _HackKernel(TileDispatcher):
    def __init__(self, cubin: bytes, func_name: str, arity: int):
        self._cubin = cubin
        self._func_name = func_name
        annotations = tuple(LeafAnnotationNode(constant=False) for _ in range(arity))
        super().__init__(annotations)

    def _compile(self, signature, ctx, compute_capability):
        return self._cubin, self._func_name, None, []
