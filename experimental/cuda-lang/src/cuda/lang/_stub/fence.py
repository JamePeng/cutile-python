# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Literal

from cuda.lang._execution import function
from .._enums import FenceProxyKind, MemoryOrder, MemoryScope, MemorySpace
from . import nvvm_mlir_interfaces as _mlir


@function
def fence_sc_cluster() -> None:
    _mlir.fence_sc_cluster()


@function
def fence_mbarrier_init() -> None:
    _mlir.fence_mbarrier_init()


@function
def fence_sync_restrict(
    order: Literal[MemoryOrder.ACQUIRE, MemoryOrder.RELEASE],
) -> None:
    _mlir.fence_sync_restrict(order=order)


@function
def fence_proxy(
    kind: FenceProxyKind,
    *,
    space: MemorySpace | None = None,
) -> None:
    _mlir.fence_proxy(kind=kind, space=space)


@function
def fence_proxy_acquire(
    address,
    size: int,
    *,
    scope: MemoryScope,
    from_proxy: FenceProxyKind = FenceProxyKind.GENERIC,
    to_proxy: FenceProxyKind = FenceProxyKind.TENSORMAP,
) -> None:
    _mlir.fence_proxy_acquire(
        addr=address,
        size=size,
        scope=scope,
        from_proxy=from_proxy,
        to_proxy=to_proxy,
    )


@function
def fence_proxy_release(
    *,
    scope: MemoryScope,
    from_proxy: FenceProxyKind = FenceProxyKind.GENERIC,
    to_proxy: FenceProxyKind = FenceProxyKind.TENSORMAP,
) -> None:
    _mlir.fence_proxy_release(
        scope=scope,
        from_proxy=from_proxy,
        to_proxy=to_proxy,
    )


@function
def fence_proxy_sync_restrict(
    order: Literal[MemoryOrder.ACQUIRE, MemoryOrder.RELEASE],
    *,
    from_proxy: FenceProxyKind = FenceProxyKind.GENERIC,
    to_proxy: FenceProxyKind = FenceProxyKind.ASYNC,
) -> None:
    _mlir.fence_proxy_sync_restrict(
        order=order,
        from_proxy=from_proxy,
        to_proxy=to_proxy,
    )


__all__ = (
    "FenceProxyKind",
    "fence_sync_restrict",
    "fence_sc_cluster",
    "fence_mbarrier_init",
    "fence_proxy_sync_restrict",
    "fence_proxy",
    "fence_proxy_acquire",
    "fence_proxy_release",
)
