# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("cuda.lang", reason="Skipping cuda-lang test: module not found")

from .util import log_ptx, no_log_ptx  # noqa: E402

__all__ = ("log_ptx", "no_log_ptx")
