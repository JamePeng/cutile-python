# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Kernel-local schedule-section tag for FMHA work bodies.

The captured schedules pass this enum explicitly as a compile-time constant so
resource methods can distinguish peeled head, steady-state loop, and tail calls.
"""

import enum


class FmhaStage(enum.Enum):
    """Schedule section of a single FMHA context work call."""

    Head = 0
    Loop = 1
    Tail = 2
