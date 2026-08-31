# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional
from enum import Enum, auto

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
from cuda.tile._memory_model import MemoryScope
from cuda.tile._ir.ir import MemoryEffect
import cuda.lang._datatype as datatype
from cuda.lang._enums import VectorReduction
from .ir import Operation, Var, attribute, operand
from .type import VectorTy, ScalarTy


@dataclass(eq=False)
class RawLLVMIntrinsic(
    Operation, opcode="llvm.call_intrinsic", memory_effect=MemoryEffect.STORE
):
    intrinsic: str = attribute()
    operands_: tuple[Var, ...] = operand()


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


@dataclass(eq=False)
class InlinePTX(Operation, opcode="inline_ptx", memory_effect=MemoryEffect.STORE):
    ptx_code: str = attribute()
    read_only_operands: tuple[Var, ...] = operand()
    write_only_operands: tuple[datatype.DType, ...] = attribute()
    read_write_operands: tuple[Var, ...] = operand()

    class RMWMode(Enum):
        READ_ONLY = auto()
        WRITE_ONLY = auto()
        READ_WRITE = auto()


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
    pointer: Var = operand()
    value: Var = operand()
    alignment: Optional[int] = attribute()


@dataclass(eq=False)
class LoadPointer(Operation, opcode="load_pointer", memory_effect=MemoryEffect.LOAD):
    pointer: Var = operand()
    alignment: Optional[int] = attribute()


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


@dataclass(eq=False)
class ReinterpretPointerAsArray(Operation, opcode="reinterpret_ptr_as_array"):
    pointer: Var = operand()


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
