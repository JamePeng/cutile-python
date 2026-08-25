# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, TYPE_CHECKING

from cuda.lang._enums import (
    CTAGroup,
    MemoryOrder,
    RoundingMode,
    SaturationMode,
    Tcgen05CopyMulticast,
    Tcgen05CopyShape,
    Tcgen05CopySourceFormat,
    TMALoadMode,
    TMAStoreMode,
)
from typing_extensions import override
from cuda.tile._memory_model import MemoryScope
from cuda.tile._ir.ir import MemoryEffect
from cuda.tile._ir.type import TensorLikeTy
from cuda.lang._enums import VectorReduction
from .ir import Operation, Var, attribute, operand
from .type import Type, VectorTy, ScalarTy, PointerTy
from .. import _llvm_bitcode as llvm
from .._passes.ir2llvm import LLVMLoweringContext, type_to_llvm

if TYPE_CHECKING:
    from cuda.lang._execution import kernel


@dataclass(eq=False)
class RawLLVMIntrinsic(
    Operation, opcode="llvm.call_intrinsic", memory_effect=MemoryEffect.STORE
):
    intrinsic: str = attribute()
    operands_: tuple[Var | None, ...] = operand()
    metadata_args: tuple[Any, ...] = attribute(default=())

    def generate_llvm(self, ctx):
        all_operands: list[llvm.Value | llvm.Metadata] = []
        operand_types = []

        meta_iter = iter(self.metadata_args)
        for x in self.operands_:
            if x is None:
                meta = next(meta_iter)
                all_operands.append(_metadata_to_llvm(meta, ctx.builder.metadata))
                operand_types.append(None)
            else:
                all_operands.append(ctx.value(x))
                operand_types.append(x.get_type())

        assert tuple(meta_iter) == ()

        return ctx.call_intrinsic(self.intrinsic,
                                  [x.get_type() for x in self.result_vars],
                                  operand_types,
                                  all_operands)


def _metadata_to_llvm(meta: Any, metadata_table: llvm.MetadataTable) -> llvm.Metadata:
    if isinstance(meta, str):
        return metadata_table.string(meta)
    else:
        raise TypeError(f"Unexpected LLVM metadata type '{type(meta)}'")


@dataclass(eq=False)
class MathUnaryOperation(Operation, opcode="math_unary"):
    fn: str = attribute()
    x: Var = operand()
    approx: bool = attribute(default=False)
    flush_to_zero: bool = attribute(default=False)


@dataclass(eq=False)
class MathBinaryOperation(Operation, opcode="math_binary"):
    fn: str = attribute()
    lhs: Var = operand()
    rhs: Var = operand()
    approx: bool = attribute(default=False)
    propagate_nan: bool = attribute(default=False)


@dataclass(eq=False)
class VectorConstruct(Operation, opcode="vector_construct"):
    elements: tuple[Var[ScalarTy], ...] = operand()


@dataclass(eq=False)
class VectorInsert(Operation, opcode="vector_insert"):
    vector: Var[VectorTy] = operand()
    value: Var[ScalarTy] = operand()
    index: Var[ScalarTy] = operand()


@dataclass(eq=False)
class CopyAsyncBulkTensorGlobalToShared(
    Operation, opcode="copy_async_bulk_tensor_g2s", memory_effect=MemoryEffect.STORE
):
    dst_memory: Var = operand()
    tensor_map: Var = operand()
    coordinates: tuple[Var, ...] = operand()
    mbarrier: Var = operand()
    im2col_offsets: tuple[Var, ...] = operand()
    multicast_mask: Var | None = operand(default=None)
    l2_cache_hint: Var | None = operand(default=None)
    predicate: Var | None = operand(default=None)
    mode: TMALoadMode = attribute()
    is_cta_only: bool = attribute()
    cta_group: CTAGroup | None = attribute(default=None)


@dataclass(eq=False)
class CopyAsyncBulkTensorSharedToGlobal(
    Operation, opcode="copy_async_bulk_tensor_s2g", memory_effect=MemoryEffect.STORE
):
    tensor_map: Var = operand()
    src_memory: Var = operand()
    coordinates: tuple[Var, ...] = operand()
    l2_cache_hint: Var | None = operand(default=None)
    predicate: Var | None = operand(default=None)
    mode: TMAStoreMode = attribute()


@dataclass(eq=False)
class Tcgen05Copy(
    Operation, opcode="tcgen05_copy", memory_effect=MemoryEffect.STORE
):
    address: Var = operand()
    shared_memory_descriptor: Var = operand()
    shape: Tcgen05CopyShape = attribute()
    cta_group: CTAGroup = attribute()
    multicast: Tcgen05CopyMulticast | None = attribute(default=None)
    source_format: Tcgen05CopySourceFormat | None = attribute(default=None)


@dataclass(frozen=True)
class InlineAsmInput:
    index: int


@dataclass(frozen=True)
class InlineAsmOutput:
    index: int


InlineAsmPiece = str | InlineAsmInput | InlineAsmOutput


@dataclass(eq=False)
class InlinePTX(Operation, opcode="inline_ptx", memory_effect=MemoryEffect.STORE):
    text: tuple[InlineAsmPiece, ...] = attribute()
    inputs: tuple[Var, ...] = operand()


@dataclass(eq=False)
class Fence(Operation, opcode="fence", memory_effect=MemoryEffect.STORE):
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()


@dataclass(eq=False)
class ForeignFunction(
    Operation, opcode="foreign_function", memory_effect=MemoryEffect.STORE
):
    function_name: str = attribute()
    operands_: tuple[Var, ...] = operand()


@dataclass(eq=False)
class VectorGetItem(
    Operation, opcode="vector_getitem", memory_effect=MemoryEffect.LOAD
):
    x: Var[VectorTy] = operand()
    index: Var[ScalarTy] = operand()


@dataclass(eq=False)
class VectorReduce(Operation, opcode="vector_reduce"):
    x: Var[VectorTy] = operand()
    kind: VectorReduction = attribute()
    propagate_nan: bool = attribute(default=False)
    reassociate: bool = attribute(default=False)


@dataclass(eq=False)
class BitCast(Operation, opcode="bitcast"):
    x: Var = operand()


@dataclass(eq=False)
class StorePointer(Operation, opcode="store_pointer", memory_effect=MemoryEffect.STORE):
    pointer: Var[PointerTy] = operand()
    value: Var[TensorLikeTy] = operand()
    alignment: Optional[int] = attribute()

    @override
    def generate_llvm(self, ctx: LLVMLoweringContext):
        value = ctx.value(self.value)
        storage_type = type_to_llvm(self.value.get_type(), ctx.builder.type_table, storage=True)
        register_type = ctx.typeof(self.value)
        if storage_type != register_type:
            # Extend i1 -> i8 for booleans etc.
            value = ctx.builder.cast(storage_type, llvm.Cast.ZEXT, value)
        pointer = ctx.value(self.pointer)
        ctx.builder.store(pointer, value, self.alignment)


@dataclass(eq=False)
class LoadPointer(Operation, opcode="load_pointer", memory_effect=MemoryEffect.LOAD):
    pointer: Var[PointerTy] = operand()
    alignment: Optional[int] = attribute()

    @override
    def generate_llvm(self, ctx: LLVMLoweringContext):
        pointer = ctx.value(self.pointer)
        storage_type = type_to_llvm(self.result_var.get_type(), ctx.builder.type_table,
                                    storage=True)
        register_type = ctx.typeof(self.result_var)
        value = ctx.builder.load(storage_type, pointer, self.alignment)
        if storage_type != register_type:
            # Truncate i8 -> i1 for booleans etc.
            value = ctx.builder.cast(register_type, llvm.Cast.TRUNC, value)
        return value


@dataclass(eq=False)
class AtomicStore(Operation, opcode="atomic_store", memory_effect=MemoryEffect.STORE):
    pointer: Var = operand()
    value: Var = operand()
    alignment: int = attribute()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()
    mmio: bool = attribute()

    VALID_MEMORY_ORDERS = (
        MemoryOrder.RELAXED,
        MemoryOrder.RELEASE,
    )
    VALID_MMIO_MEMORY_ORDERS = (
        MemoryOrder.RELAXED,
        MemoryOrder.RELEASE,
    )


@dataclass(eq=False)
class AtomicLoad(Operation, opcode="atomic_load", memory_effect=MemoryEffect.LOAD):
    pointer: Var = operand()
    alignment: int = attribute()
    memory_order: MemoryOrder = attribute()
    memory_scope: MemoryScope = attribute()
    mmio: bool = attribute()

    VALID_MEMORY_ORDERS = (
        MemoryOrder.RELAXED,
        MemoryOrder.ACQUIRE,
    )
    VALID_MMIO_MEMORY_ORDERS = (
        MemoryOrder.RELAXED,
        MemoryOrder.ACQUIRE,
    )

    @property
    def has_observable_effect(self) -> bool:
        return self.mmio or self.memory_order is MemoryOrder.ACQUIRE


@dataclass
class TensorMapAsOpaquePtr(Operation, opcode="tensor_map_as_opaque_ptr"):
    tensor_map: Var = operand()


@dataclass(eq=False)
class FmaOperation(Operation, opcode="fma"):
    x: Var = operand()
    y: Var = operand()
    z: Var = operand()
    rounding_mode: RoundingMode = attribute()
    saturation_mode: SaturationMode = attribute()
    flush_to_zero: bool = attribute()
    relu: bool = attribute()
    oob: bool = attribute()


@dataclass(eq=False)
class KernelLaunch(
    Operation, opcode="kernel_launch", memory_effect=MemoryEffect.STORE
):
    stream: Var = operand()
    block_count: tuple[Var, ...] = operand()
    thread_count: tuple[Var, ...] = operand()
    kernel_argument_leaves: tuple[Var, ...] = operand()
    kernel_argument_types: tuple[Type, ...] = attribute()
    launched_kernel: kernel = attribute()
    cooperative: bool = attribute()
    block_in_cluster_count: tuple[Var, ...] | None = operand()
    preferred_block_in_cluster_count: tuple[Var, ...] | None = operand()
    programmatic_dependent_launch: bool = attribute()
