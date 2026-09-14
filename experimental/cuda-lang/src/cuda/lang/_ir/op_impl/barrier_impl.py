# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang._datatype as datatype
from cuda.lang._ir.type import ScalarTy
from cuda.lang._ir.ir import add_operation
from cuda.tile._ir.ir import add_operation_variadic
from cuda.lang._ir.op_defs import RawLLVMIntrinsic
from cuda.tile._ir.ops import implicit_cast
from cuda.tile._ir.op_impl import ImplRegistry, require_constant_enum
from cuda.lang._ir.type_checking_helpers import (
    require_integral_scalar_type,
    require_boolean_scalar_type,
    optional_cast,
)
from cuda.lang._stub import barrier
from cuda.lang._exception import TypeCheckingError

_registry = ImplRegistry()
impl = _registry.impl


def barrier_impl_registry() -> ImplRegistry:
    return _registry


def barrier_sync_block(number_of_threads, barrier_id, aligned: bool):
    require_integral_scalar_type(barrier_id)
    barrier_id = implicit_cast(barrier_id, datatype.int32, "barrier id")
    number_of_threads = optional_cast(
        number_of_threads, datatype.int32, "barrier number_of_threads"
    )
    intrinsic = "llvm.nvvm.barrier.cta.sync"
    if aligned:
        intrinsic += ".aligned"
    if number_of_threads is None:
        intrinsic += ".all"
        operands = (barrier_id,)
    else:
        intrinsic += ".count"
        operands = (barrier_id, number_of_threads)

    add_operation_variadic(
        RawLLVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=operands,
    )


@impl(barrier.barrier_sync_block)
def barrier_sync_block_impl(number_of_threads, barrier_id):
    return barrier_sync_block(number_of_threads, barrier_id, False)


@impl(barrier.barrier_sync_block_aligned)
def barrier_sync_block_aligned_impl(number_of_threads, barrier_id):
    return barrier_sync_block(number_of_threads, barrier_id, True)


def barrier_arrive_cluster(memory_order, aligned: bool):
    memory_order = require_constant_enum(memory_order, barrier.MemoryOrder)
    if memory_order not in (barrier.MemoryOrder.RELEASE, barrier.MemoryOrder.RELAXED):
        raise TypeCheckingError(
            "barrier_arrive_cluster memory_order must be "
            "MemoryOrder.RELEASE or MemoryOrder.RELAXED"
        )
    intrinsic = "llvm.nvvm.barrier.cluster.arrive"
    if memory_order is barrier.MemoryOrder.RELAXED:
        intrinsic += ".relaxed"
    if aligned:
        intrinsic += ".aligned"

    add_operation_variadic(
        RawLLVMIntrinsic,
        (),
        intrinsic=intrinsic,
        operands_=(),
    )


@impl(barrier.barrier_arrive_cluster)
def barrier_arrive_cluster_impl(memory_order):
    return barrier_arrive_cluster(memory_order, False)


@impl(barrier.barrier_arrive_cluster_aligned)
def barrier_arrive_cluster_aligned_impl(memory_order):
    return barrier_arrive_cluster(memory_order, True)


def barrier_reduce_block(
    op,
    predicate,
    number_of_threads,
    barrier_id,
    aligned: bool,
):
    op = require_constant_enum(op, barrier.BarrierReductionKind)
    require_boolean_scalar_type(predicate)
    require_integral_scalar_type(barrier_id)
    barrier_id = implicit_cast(barrier_id, datatype.int32, "barrier id")
    number_of_threads = optional_cast(
        number_of_threads, datatype.int32, "barrier number_of_threads"
    )
    intrinsic = "llvm.nvvm.barrier.cta.red."
    match op:
        case barrier.BarrierReductionKind.POP_COUNT:
            intrinsic += "popc"
        case barrier.BarrierReductionKind.AND:
            intrinsic += "and"
        case barrier.BarrierReductionKind.OR:
            intrinsic += "or"
        case _:
            assert False

    if aligned:
        intrinsic += ".aligned"
    if number_of_threads is None:
        intrinsic += ".all"
        operands = (barrier_id, predicate)
    else:
        intrinsic += ".count"
        operands = (barrier_id, number_of_threads, predicate)

    result_type = (
        ScalarTy(datatype.int32)
        if op is barrier.BarrierReductionKind.POP_COUNT
        else ScalarTy(datatype.bool_)
    )
    return add_operation(
        RawLLVMIntrinsic,
        result_type,
        intrinsic=intrinsic,
        operands_=operands,
    )


@impl(barrier.barrier_reduce_block)
def barrier_reduce_block_impl(
    op,
    predicate,
    number_of_threads,
    barrier_id,
):
    return barrier_reduce_block(
        op, predicate, number_of_threads, barrier_id, False
    )


@impl(barrier.barrier_reduce_block_aligned)
def barrier_reduce_block_aligned_impl(
    op,
    predicate,
    number_of_threads,
    barrier_id,
):
    return barrier_reduce_block(
        op, predicate, number_of_threads, barrier_id, True
    )
