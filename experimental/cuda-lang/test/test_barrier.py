# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl
from cuda.lang._exception import CompilerExecutionError
from .util import compile_kernel
import pytest

HOPPER_TARGET = {"gpu_name": "sm_90", "arch": "compute_90"}


def barrier_sync_block_cases():
    for op, aligned in (
        (cl.barrier_sync_block, False),
        (cl.barrier_sync_block_aligned, True),
    ):
        for number_of_threads in (None, 5, 32):
            if number_of_threads == 5:
                raises = pytest.raises(
                    CompilerExecutionError,
                    match=(
                        "Number of threads participating in barrier must be "
                        "in multiple of warp size"
                    ),
                )
                yield op, number_of_threads, None, raises
            else:
                ptx = ("bar" if aligned else "barrier") + ".sync"
                yield op, number_of_threads, ptx, None


@pytest.mark.parametrize(
    "op, number_of_threads, expect, raises", barrier_sync_block_cases()
)
def test_barrier_sync_block(op, number_of_threads, expect, raises):
    def kernel():
        op(number_of_threads, 7)

    compile_kernel(kernel, assert_in_ptx=expect, raises=raises)


def barrier_arrive_block_cases():
    for op, aligned in (
        (cl.barrier_arrive_block, False),
        (cl.barrier_arrive_block_aligned, True),
    ):
        for number_of_threads in (None, 5, 32):
            if number_of_threads is None:
                raises = pytest.raises(
                    Exception,
                    match="Expected a scalar value, but given value has type None",
                )
                yield op, number_of_threads, None, raises
            elif number_of_threads == 5:
                raises = pytest.raises(
                    CompilerExecutionError,
                    match=(
                        "Number of threads participating in barrier must "
                        "be in multiple of warp size"
                    ),
                )
                yield op, number_of_threads, None, raises
            else:
                ptx = ("bar" if aligned else "barrier") + ".arrive"
                yield op, number_of_threads, ptx, None


@pytest.mark.parametrize(
    "op, number_of_threads, expect, raises", barrier_arrive_block_cases()
)
def test_barrier_arrive_block(op, number_of_threads, expect, raises):
    def kernel():
        op(number_of_threads, 7)

    compile_kernel(kernel, assert_in_ptx=expect, raises=raises)


def barrier_reduce_block_cases():
    for barrier_op, aligned in (
        (cl.barrier_reduce_block, False),
        (cl.barrier_reduce_block_aligned, True),
    ):
        for reduction_op in (*cl.BarrierReductionKind, None, 5):
            for predicate in (True, False, None, 5):
                for number_of_threads in (None, 5, 32):
                    if not isinstance(reduction_op, cl.BarrierReductionKind):
                        raises = pytest.raises(Exception, match="BarrierReductionKind")
                        yield barrier_op, reduction_op, predicate, number_of_threads, None, raises
                    elif not isinstance(predicate, bool):
                        raises = pytest.raises(
                            Exception,
                            match="Expected (a scalar|boolean scalar)",
                        )
                        yield barrier_op, reduction_op, predicate, number_of_threads, None, raises
                    elif number_of_threads == 5:
                        raises = pytest.raises(
                            CompilerExecutionError,
                            match=(
                                "Number of threads participating in barrier "
                                "must be in multiple of warp size"
                            ),
                        )
                        yield barrier_op, reduction_op, predicate, number_of_threads, None, raises
                    else:
                        ptx_op = {
                            cl.BarrierReductionKind.POP_COUNT: "popc",
                            cl.BarrierReductionKind.AND: "and",
                            cl.BarrierReductionKind.OR: "or",
                        }[reduction_op]
                        ptx = ("bar" if aligned else "barrier") + f".red.{ptx_op}"
                        yield barrier_op, reduction_op, predicate, number_of_threads, ptx, None


@pytest.mark.parametrize(
    "barrier_op, reduction_op, predicate, number_of_threads, expect, raises",
    barrier_reduce_block_cases(),
)
def test_barrier_reduce_block(
    barrier_op, reduction_op, predicate, number_of_threads, expect, raises
):
    def kernel():
        barrier_op(reduction_op, predicate, number_of_threads, 7)

    compile_kernel(kernel, assert_in_ptx=expect, raises=raises)


@pytest.mark.parametrize(
    "op, expect",
    (
        ("POP_COUNT", "bar.red.popc"),
        ("AND", "bar.red.and"),
        ("OR", "bar.red.or"),
    ),
)
def test_barrier_reduce_block_accepts_enum_member_name(op, expect):
    def kernel():
        cl.barrier_reduce_block_aligned(op, True)

    compile_kernel(kernel, assert_in_ptx=expect)


def barrier_arrive_cluster_cases():
    valid_orders = (cl.MemoryOrder.RELEASE, cl.MemoryOrder.RELAXED)
    for op, aligned in (
        (cl.barrier_arrive_cluster, False),
        (cl.barrier_arrive_cluster_aligned, True),
    ):
        for order in (*tuple(cl.MemoryOrder), 5, None):
            if order not in tuple(cl.MemoryOrder):
                raises = pytest.raises(Exception, match="MemoryOrder")
                yield op, order, None, raises
            elif order not in valid_orders:
                raises = pytest.raises(
                    Exception,
                    match="memory_order must be MemoryOrder.RELEASE or MemoryOrder.RELAXED",
                )
                yield op, order, None, raises
            else:
                expect = "barrier.cluster.arrive"
                expect += ".relaxed" if order == cl.MemoryOrder.RELAXED else ""
                expect += ".aligned" if aligned else ""
                yield op, order, expect, None


@pytest.mark.parametrize("op, order, expect, raises", barrier_arrive_cluster_cases())
def test_barrier_arrive_cluster(op, order, expect, raises):
    def kernel():
        op(memory_order=order)

    compile_kernel(
        kernel,
        assert_in_ptx=expect,
        raises=raises,
        **HOPPER_TARGET,
    )


def test_barrier_arrive_cluster_accepts_enum_member_name():
    def kernel():
        cl.barrier_arrive_cluster_aligned(memory_order="RELAXED")

    compile_kernel(
        kernel,
        assert_in_ptx="barrier.cluster.arrive.relaxed.aligned",
        **HOPPER_TARGET,
    )


@pytest.mark.parametrize(
    "op, expect",
    (
        (cl.barrier_wait_cluster, "barrier.cluster.wait"),
        (cl.barrier_wait_cluster_aligned, "barrier.cluster.wait.aligned"),
    ),
)
def test_barrier_wait_cluster(op, expect):
    def kernel():
        op()

    compile_kernel(
        kernel,
        assert_in_ptx=expect,
        **HOPPER_TARGET,
    )


@pytest.mark.parametrize(
    "op, expect",
    (
        (
            cl.barrier_sync_cluster,
            ("barrier.cluster.arrive", "barrier.cluster.wait"),
        ),
        (
            cl.barrier_sync_cluster_aligned,
            ("barrier.cluster.arrive.aligned", "barrier.cluster.wait.aligned"),
        ),
    ),
)
def test_barrier_sync_cluster(op, expect):
    def kernel():
        op()

    compile_kernel(
        kernel,
        assert_in_ptx=expect,
        **HOPPER_TARGET,
    )


def test_barrier_sync_warp():
    def kernel():
        cl.barrier_sync_warp(32)

    compile_kernel(kernel, assert_in_ptx="bar.warp.sync")


def test_syncthreads():
    def kernel():
        cl.syncthreads()

    compile_kernel(kernel, assert_in_ptx="bar.sync")


def test_syncwarp():
    def kernel():
        cl.syncwarp(32)

    compile_kernel(kernel, assert_in_ptx="bar.warp.sync")


@pytest.mark.parametrize(
    "op, expected_ptx, expected_dtype", ((cl.syncthreads_count, "bar.red.popc", cl.int32),
                                         (cl.syncthreads_and, "bar.red.and", cl.bool_),
                                         (cl.syncthreads_or, "bar.red.or", cl.bool_)))
def test_syncthreads_count(op, expected_ptx, expected_dtype):
    def kernel():
        res = op(True)
        cl.static_assert(cl.dtype_of(res) == expected_dtype)

    compile_kernel(kernel, assert_in_ptx=expected_ptx)
