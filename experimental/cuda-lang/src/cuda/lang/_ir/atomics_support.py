# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.tile._ir.arithmetic_ops import astype
from cuda.tile._ir.op_impl import require_constant_enum
from cuda.lang._enums import AtomicOp, MemoryOrder
from cuda.tile._memory_model import MemoryScope

import cuda.lang._datatype as datatype
from cuda.lang._exception import TypeCheckingError
from .ir import Operation, Var
from .type import PointerTy, ScalarTy
from .type_checking_helpers import require_scalar_type


ATOMIC_ADD_DTYPES = (
    datatype.int32,
    datatype.uint32,
    datatype.int64,
    datatype.uint64,
    datatype.float16,
    datatype.bfloat16,
    datatype.float32,
    datatype.float64,
)
ATOMIC_SUB_DTYPES = (
    datatype.int32,
    datatype.uint32,
    datatype.int64,
    datatype.uint64,
    datatype.float32,
    datatype.float64,
)
ATOMIC_BITWISE_DTYPES = (
    datatype.int32,
    datatype.uint32,
    datatype.int64,
    datatype.uint64,
)
ATOMIC_MIN_MAX_DTYPES = (
    datatype.int32,
    datatype.uint32,
    datatype.int64,
    datatype.uint64,
    datatype.float32,
    datatype.float64,
)
ATOMIC_INC_DEC_DTYPES = (datatype.uint32,)
ATOMIC_XCHG_DTYPES = (
    datatype.int32,
    datatype.uint32,
    datatype.float32,
    datatype.int64,
    datatype.uint64,
    datatype.float64,
)
ATOMIC_CAS_DTYPES = (
    datatype.int16,
    datatype.uint16,
    datatype.int32,
    datatype.uint32,
    datatype.int64,
    datatype.uint64,
)

ATOMIC_RMW_SUPPORTED_DTYPES = {
    AtomicOp.ADD: ATOMIC_ADD_DTYPES,
    AtomicOp.SUB: ATOMIC_SUB_DTYPES,
    AtomicOp.AND: ATOMIC_BITWISE_DTYPES,
    AtomicOp.OR: ATOMIC_BITWISE_DTYPES,
    AtomicOp.XOR: ATOMIC_BITWISE_DTYPES,
    AtomicOp.MIN: ATOMIC_MIN_MAX_DTYPES,
    AtomicOp.MAX: ATOMIC_MIN_MAX_DTYPES,
    AtomicOp.INC: ATOMIC_INC_DEC_DTYPES,
    AtomicOp.DEC: ATOMIC_INC_DEC_DTYPES,
}

ATOMIC_VALID_MEMORY_ORDERS = (
    MemoryOrder.RELAXED,
    MemoryOrder.ACQUIRE,
    MemoryOrder.RELEASE,
    MemoryOrder.ACQ_REL,
)
ATOMIC_VALID_MEMORY_SCOPES = (
    MemoryScope.BLOCK,
    MemoryScope.CLUSTER,
    MemoryScope.DEVICE,
    MemoryScope.SYS,
)


def format_supported_dtypes(dtypes: tuple[datatype.DType, ...]) -> str:
    return ", ".join(str(dtype) for dtype in dtypes)


def require_atomic_dtype(
    op: AtomicOp, dtype: datatype.DType, supported_dtypes: tuple[datatype.DType, ...]
):
    if dtype not in supported_dtypes:
        raise TypeCheckingError(
            f"{op.value} does not support dtype {dtype}; supported dtypes are "
            f"{format_supported_dtypes(supported_dtypes)}"
        )


def require_atomic_memory_order_and_scope(
    operation_type: type[Operation],
    memory_order_var: Var,
    memory_scope_var: Var,
    mmio: bool = False,
) -> tuple[MemoryOrder, MemoryScope]:
    memory_order = require_constant_enum(memory_order_var, MemoryOrder)
    memory_scope = require_constant_enum(memory_scope_var, MemoryScope)

    valid_memory_orders = (
        operation_type.VALID_MMIO_MEMORY_ORDERS
        if mmio
        else operation_type.VALID_MEMORY_ORDERS
    )
    if memory_order not in valid_memory_orders:
        expected = ", ".join(str(order) for order in valid_memory_orders)
        raise TypeCheckingError(
            f"Invalid memory order for {operation_type._opcode}. "
            f"Got {memory_order}, expected one of {expected}"
        )

    if memory_scope not in ATOMIC_VALID_MEMORY_SCOPES:
        expected = ", ".join(str(scope) for scope in ATOMIC_VALID_MEMORY_SCOPES)
        raise TypeCheckingError(
            f"Invalid memory scope for {operation_type._opcode}. "
            f"Got {memory_scope}, expected one of {expected}"
        )

    return memory_order, memory_scope


def require_atomic_rmw_value(
    op: AtomicOp, ptr_ty: PointerTy, val: Var
) -> tuple[Var, ScalarTy]:
    require_scalar_type(val)
    ptr_dtype = ptr_ty.pointee_dtype
    require_atomic_dtype(op, ptr_dtype, ATOMIC_RMW_SUPPORTED_DTYPES[op])
    return astype(val, ptr_dtype), ScalarTy(ptr_dtype)
