# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lower native CUDA Lang host functions to MLIR."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from cuda.tile import _datatype as datatype
from cuda.tile._ir.ir import Var

from cuda.lang import _mlir as mlir
from cuda.lang._exception import TypeCheckingError
from cuda.lang._ir import ir, ops
from cuda.lang._ir.type import PointerTy, ScalarTy
import cuda.lang._mlir.extras.types as T
from cuda.lang._passes.ir2mlir.pass_definition import (
    MLIRLoweringContext,
    create_mlir_blocks,
    lower_mlir_region,
    mlir_op_lowering,
)
from cuda.lang._passes.ir2mlir.type_conversion import (
    ir_type_to_mlir_type,
    mlir_constant_of_type,
)


_HOST_ENTRY_SYMBOL = "cuda_lang_host_entry"
_HOST_ENTRY_PARAMETER_NAMES = ("abi_arguments",)


@dataclass(frozen=True)
class _HostEntryABI:
    """Ordered native ABI for the physical compiled-host entry block."""

    function_type: mlir.llvm.LLVMFunctionType
    parameter_names: tuple[str, ...]

    @property
    def parameter_types(self) -> tuple[mlir.Type, ...]:
        return tuple(self.function_type.params)

    def __post_init__(self) -> None:
        if len(self.parameter_names) != len(self.parameter_types):
            raise ValueError("host entry ABI parameter names and types must match")


@dataclass(frozen=True)
class _HostRuntimeArguments:
    """Physical arguments supplied by the compiled-host runtime."""

    abi_arguments: mlir.Value


@dataclass(kw_only=True)
class HostLoweringContext(MLIRLoweringContext):
    """Data required while lowering a native CUDA Lang host function."""

    abi_address_by_slot: dict[int, mlir.Value] = field(default_factory=dict)
    physical_entry_block: mlir.Block | None = None
    runtime_arguments: _HostRuntimeArguments | None = None
    host_entry_function_type: mlir.llvm.LLVMFunctionType | None = None
    pointer_type: mlir.llvm.LLVMPointerType = field(
        default_factory=mlir.llvm.LLVMPointerType
    )
    print_format_index: int = 0
    printf_declared: bool = False


class HostIR2MLIR:
    """Build a native host function using shared operation lowering."""

    def __init__(self, region: ir.Region, ctx: ir.IRContext):
        self.context = HostLoweringContext(region=region, ir_context=ctx)
        entry_type = mlir.llvm.LLVMFunctionType(
            returnType=T.i32(),
            params=(self.context.pointer_type,),
            varArg=False,
        )
        self.context.host_entry_function_type = entry_type
        self.host_entry_abi = _HostEntryABI(
            function_type=entry_type,
            parameter_names=_HOST_ENTRY_PARAMETER_NAMES,
        )

    def __call__(self) -> mlir.Operation:
        self.setup_func_op()
        self.setup_blocks()
        self.lower_physical_entry()
        lower_mlir_region(self.context)
        assert self.context.module_op is not None
        return self.context.module_op

    def setup_func_op(self) -> None:
        context = self.context
        with mlir.Block().append_here() as top_block:
            module_region = mlir.Region()
            mlir.add_ModuleOp(bodyRegion=module_region)
        context.module_op = top_block[0]

        with module_region.new_block().append_here() as module_block:
            body_region = mlir.Region()
            mlir.llvm.add_LLVMFuncOp(
                sym_name=_HOST_ENTRY_SYMBOL,
                function_type=self.host_entry_abi.function_type,
                body=body_region,
            )
        context.func_op = module_block.operations[-1]

    def setup_blocks(self) -> None:
        context = self.context
        physical_args = tuple(
            mlir.Value(parameter_type, parameter_name)
            for parameter_type, parameter_name in zip(
                self.host_entry_abi.parameter_types,
                self.host_entry_abi.parameter_names,
                strict=True,
            )
        )
        context.physical_entry_block = context.function_region.new_block(
            args=physical_args,
            block_id="physical_entry",
        )
        context.runtime_arguments = _HostRuntimeArguments(
            **dict(zip(self.host_entry_abi.parameter_names, physical_args, strict=True))
        )
        create_mlir_blocks(context)

    def lower_physical_entry(self) -> None:
        context = self.context
        physical_entry = context.physical_entry_block
        assert physical_entry is not None
        logical_entry = context.region.blocks[0]
        with physical_entry.append_here():
            arguments = tuple(
                _load_abi_argument(context, parameter, slot)
                for slot, parameter in enumerate(logical_entry.params)
            )
            mlir.cf.add_BranchOp(
                dest=context.block_map[logical_entry].label,
                destOperands=arguments,
            )


@mlir_op_lowering(device=False)
def lower_return(
    context: HostLoweringContext, operation: ops.Return
) -> Sequence[mlir.Value]:
    assert operation.result_vars == (), "compiled host returns must not produce values"
    entry_type = context.host_entry_function_type
    assert entry_type is not None
    success = mlir_constant_of_type(entry_type.returnType, 0)
    mlir.llvm.add_ReturnOp(arg=success)
    return ()


def _abi_argument_address(
    context: HostLoweringContext, slot: int
) -> mlir.Value:
    if slot in context.abi_address_by_slot:
        return context.abi_address_by_slot[slot]
    runtime_arguments = context.runtime_arguments
    assert runtime_arguments is not None
    address_slot = mlir.llvm.add_GEPOp(
        res_type=context.pointer_type,
        base=runtime_arguments.abi_arguments,
        dynamicIndices=(),
        rawConstantIndices=(slot,),
        elem_type=context.pointer_type,
    )
    address = mlir.llvm.add_LoadOp(
        res_type=context.pointer_type, addr=address_slot
    )
    context.abi_address_by_slot[slot] = address
    return address


def _load_abi_argument(
    context: HostLoweringContext, value: Var, slot: int
) -> mlir.Value:
    address = _abi_argument_address(context, slot)
    ty = value.get_type()
    if isinstance(ty, PointerTy):
        return mlir.llvm.add_LoadOp(
            res_type=ir_type_to_mlir_type(ty), addr=address
        )
    if not isinstance(ty, ScalarTy):
        raise TypeCheckingError(
            f"host ABI value of type {ty} is not supported", loc=value.loc
        )
    return mlir.llvm.add_LoadOp(
        res_type=ir_type_to_mlir_type(ty), addr=address
    )


def _promote_printf_argument(
    context: HostLoweringContext, argument: Var
) -> mlir.Value:
    """Default argument promotion for C variadic function"""
    ty = argument.get_type()
    value = context.get_var(argument)
    if isinstance(ty, PointerTy):
        return value
    if not isinstance(ty, ScalarTy):
        raise TypeCheckingError(
            f"host printf does not support argument type {ty}",
            loc=argument.loc,
        )
    dtype = ty.dtype
    if datatype.is_boolean(dtype):
        return mlir.arith.add_ExtUIOp(out_type=T.i32(), in_=value)
    if datatype.is_integral(dtype) and dtype.bitwidth < 32:
        converter = (
            mlir.arith.add_ExtSIOp
            if datatype.is_signed(dtype)
            else mlir.arith.add_ExtUIOp
        )
        return converter(out_type=T.i32(), in_=value)
    if datatype.is_float(dtype) and dtype.bitwidth < 64:
        return mlir.arith.add_ExtFOp(out_type=T.f64(), in_=value)
    return value


@mlir_op_lowering(device=False)
def lower_printf(
    context: HostLoweringContext, operation: ops.TilePrintf
) -> Sequence[mlir.Value]:
    assert context.module_op is not None

    # Store format string as a module global constant
    encoded_format = operation.format.encode() + b"\0"
    symbol = f"cuda_lang_host_format_{context.print_format_index}"
    context.print_format_index += 1
    with context.module_op.regions[0].blocks[0].prepend_here():
        mlir.llvm.add_GlobalOp(
            global_type=mlir.llvm.LLVMArrayType(
                elementType=T.i8(),
                numElements=len(encoded_format),
            ),
            constant=True,
            sym_name=symbol,
            linkage=mlir.llvm.Linkage.Internal,
            value=mlir.StringAttr(value=encoded_format.decode()),
            visibility_=mlir.llvm.Visibility.Default,
            initializer=mlir.Region(),
        )
    # Take address of the format string
    format_address = mlir.llvm.add_AddressOfOp(
        res_type=context.pointer_type,
        global_name=symbol,
    )
    # Get the printf function address
    function_type = mlir.llvm.LLVMFunctionType(
        returnType=T.i32(),
        params=(context.pointer_type,),
        varArg=True,
    )
    if not context.printf_declared:
        with context.module_op.regions[0].blocks[0].prepend_here():
            mlir.llvm.add_LLVMFuncOp(
                sym_name="printf",
                linkage=mlir.llvm.Linkage.External,
                body=mlir.Region(),
                function_type=function_type,
            )
        context.printf_declared = True
    mlir.llvm.add_CallOp(
        result_type=T.i32(),
        var_callee_type=function_type,
        callee="printf",
        callee_operands=(
            format_address,
            *(_promote_printf_argument(context, arg) for arg in operation.args),
        ),
        op_bundle_operands=(),
        op_bundle_sizes=(),
    )
    return [None]
