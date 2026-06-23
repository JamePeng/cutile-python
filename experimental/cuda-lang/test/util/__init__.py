# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuda.lang._compile import get_compute_capability
from cuda.lang._logging import get_log_flags

from .filecheck_utils import filecheck, get_source
from .ir_utils import (
    get_ir,
    make_symbolic_scalar,
    make_symbolic_tensor,
    compile_for_arguments,
)


def require_blackwell_or_newer():
    return pytest.mark.skipif(
        get_compute_capability() < (10, 0),
        reason="feature requires Blackwell or newer",
    )


def require_blackwell_cc100():
    cc = get_compute_capability()
    return pytest.mark.skipif(
        cc.major != 10,
        reason="feature requires Blackwell with compute capability 100",
    )


def require_hopper_or_newer():
    return pytest.mark.skipif(
        get_compute_capability() < (9, 0),
        reason="feature requires Hopper or newer",
    )


@pytest.fixture
def log_ptx():
    log_flags = get_log_flags()
    old_log_ptx = log_flags.log_ptx
    log_flags.log_ptx = True
    try:
        yield
    finally:
        log_flags.log_ptx = old_log_ptx


@pytest.fixture
def no_log_ptx():
    log_flags = get_log_flags()
    old_log_ptx = log_flags.log_ptx
    log_flags.log_ptx = False
    try:
        yield
    finally:
        log_flags.log_ptx = old_log_ptx


__all__ = (
    "filecheck",
    "get_source",
    "get_ir",
    "make_symbolic_scalar",
    "make_symbolic_tensor",
    "compile_for_arguments",
    "log_ptx",
    "require_hopper_or_newer",
    "require_blackwell_or_newer",
)
