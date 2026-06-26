# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from cuda.tile import static_assert

from cuda.lang._execution import function, stub
from .._enums import BarrierReductionKind, MemoryOrder
from .core_api import FULL_MASK
from . import nvvm as _nvvm


def _require_constant_bool(var):
    static_assert(
        var in (True, False),
        f"Expected constant of type bool but got {var}",
    )


def _require_constant_enum(var, enum):
    static_assert(
        var in tuple(enum),
        f"Expected enum constant of type {enum.__name__} but got {var}",
    )


@function
def barrier_sync_block(
    number_of_threads: int | None = None,
    barrier_id: int = 0,
    *,
    aligned: bool = True,
) -> None:
    _require_constant_bool(aligned)
    if number_of_threads is None:
        if aligned:
            _nvvm.barrier_cta_sync_aligned_all(barrier_id)
        else:
            _nvvm.barrier_cta_sync_all(barrier_id)
    else:
        if aligned:
            _nvvm.barrier_cta_sync_aligned_count(barrier_id, number_of_threads)
        else:
            _nvvm.barrier_cta_sync_count(barrier_id, number_of_threads)


@function
def barrier_arrive_block(
    number_of_threads: int,
    barrier_id: int = 0,
    *,
    aligned: bool = True,
) -> None:
    _require_constant_bool(aligned)
    if aligned:
        _nvvm.barrier_cta_arrive_aligned_count(barrier_id, number_of_threads)
    else:
        _nvvm.barrier_cta_arrive_count(barrier_id, number_of_threads)


@stub
def barrier_reduce_block(
    op: BarrierReductionKind,
    predicate: bool,
    number_of_threads: int | None = None,
    barrier_id: int = 0,
    *,
    aligned: bool = True,
) -> int | bool: ...


@function
def barrier_arrive_cluster(
    *,
    aligned: bool = True,
    memory_order: Literal[
        MemoryOrder.RELEASE, MemoryOrder.RELAXED
    ] = MemoryOrder.RELEASE,
) -> None:
    _require_constant_bool(aligned)
    _require_constant_enum(memory_order, MemoryOrder)
    if memory_order == MemoryOrder.RELAXED:
        if aligned:
            _nvvm.barrier_cluster_arrive_relaxed_aligned()
        else:
            _nvvm.barrier_cluster_arrive_relaxed()
    else:
        if aligned:
            _nvvm.barrier_cluster_arrive_aligned()
        else:
            _nvvm.barrier_cluster_arrive()


@function
def barrier_wait_cluster(*, aligned: bool = True) -> None:
    _require_constant_bool(aligned)
    if aligned:
        _nvvm.barrier_cluster_wait_aligned()
    else:
        _nvvm.barrier_cluster_wait()


@function
def barrier_sync_cluster(*, aligned: bool = True) -> None:
    barrier_arrive_cluster(aligned=aligned)
    barrier_wait_cluster(aligned=aligned)


@function
def barrier_sync_warp(mask: int = FULL_MASK) -> None:
    '''
    Synchronize warp lanes selected by ``mask``.
    '''
    _nvvm.bar_warp_sync(mask)


__all__ = (
    "BarrierReductionKind",
    "barrier_sync_warp",
    "barrier_sync_block",
    "barrier_arrive_block",
    "barrier_reduce_block",
    "barrier_arrive_cluster",
    "barrier_wait_cluster",
    "barrier_sync_cluster",
)
