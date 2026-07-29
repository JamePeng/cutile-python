# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compile CUDA Lang host function."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import subprocess
import tempfile
from types import FunctionType
from typing import Protocol

from cuda.tile._annotated_function import get_annotated_function
from cuda.tile._compile import _create_kernel_parameters
from cuda.tile._passes.ast2hir import HirMode
from cuda.tile._passes.dce import dead_code_elimination_pass
from cuda.tile._passes.eliminate_assign_ops import eliminate_assign_ops
from cuda.tile._passes.hir2ir import hir2ir

from cuda.lang._compile import get_compiler_binary_path
from cuda.lang._exception import CompilerExecutionError
from cuda.lang._ir import ir
from cuda.lang._ir.ops import cuda_lang_impl_registry
from cuda.lang._passes.ast2hir import get_function_hir
from cuda.lang._passes.flatten_cfg import flatten_cfg
from cuda.lang._passes.ir2mlir.host import HostIR2MLIR
from cuda.lang.compilation import KernelSignature


class _LoadedHostCode(Protocol):
    @property
    def entry_address(self) -> int: ...


@dataclass(frozen=True)
class HostCompilation:
    """Compiler output for one typed host function."""

    signature: KernelSignature
    host_ir: ir.Block | None
    host_mlir: str | None
    _loaded_host_code: _LoadedHostCode = field(repr=False, compare=False)

    @property
    def entry_address(self) -> int:
        return self._loaded_host_code.entry_address


def _compile_native_host(mlir_text: str) -> _LoadedHostCode:
    from cuda.lang import _host_jit

    # mlir2cubin owns MLIR-to-native code generation.
    # The host_jit extension only links and loads the resulting object.
    executable = get_compiler_binary_path()
    with tempfile.TemporaryDirectory(prefix="cuda-lang-host-") as directory:
        object_path = os.path.join(directory, "host.o")
        argv = [
            executable,
            "-",
            "-o",
            object_path,
            # Required by mlir2cubin's shared command-line interface, but
            # unused when emitting a native host object.
            "--gpu-name=unused",
            "--arch=unused",
            "--emit-host-object",
        ]
        try:
            subprocess.run(
                argv, input=mlir_text.encode(), capture_output=True, check=True
            )
        except subprocess.CalledProcessError as error:
            raise CompilerExecutionError(
                return_code=error.returncode,
                stderr=error.stderr.decode(),
                compiler_flags=argv,
                compiler_version=None,
            ) from None
        with open(object_path, "rb") as object_file:
            return _host_jit.load_object(object_file.read())


def _compile(
    function: FunctionType,
    signature: KernelSignature,
    *,
    keep_ir: bool = False,
    keep_mlir: bool = False,
) -> HostCompilation:

    constraints = signature.parameters
    annotated = get_annotated_function(function)
    if len(annotated.pysig.parameters) != len(constraints):
        raise TypeError("host signature must contain every host function argument")

    host_hir = get_function_hir(function, mode=HirMode.ENTRY_POINT)
    parameter_names = tuple(host_hir.signature.parameters)
    ctx = ir.IRContext(execution_space="host")
    with (
        ir.TileBuilder(ctx, host_hir.body.loc) as builder,
        cuda_lang_impl_registry.as_current(),
    ):
        parameters = _create_kernel_parameters(
            constraints,
            annotated.parameter_annotations,
            parameter_names,
            host_hir.param_locs,
            ctx,
        )
        hir2ir(host_hir, parameters.aggregate_vars, ctx)
    host_body = ctx.make_block("host_entry", host_hir.body.loc)
    host_body.params = sum(
        (leaves for leaves, _ in parameters.nonconstant_flat_vars), ()
    )
    host_body.extend(builder.ops)

    eliminate_assign_ops(host_body)
    dead_code_elimination_pass(host_body)
    host_cfg = flatten_cfg(host_body, ctx)
    host_module = HostIR2MLIR(host_cfg, ctx)()
    host_mlir = str(host_module)
    loaded_host_code = _compile_native_host(host_mlir)
    return HostCompilation(
        signature,
        host_body if keep_ir else None,
        host_mlir if keep_mlir else None,
        loaded_host_code,
    )


__all__ = ("HostCompilation",)
