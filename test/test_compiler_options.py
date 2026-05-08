# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


import cuda.tile as ct
import pytest


def test_invalid_target_name():
    err = r"Invalid GPU architecture name: sm100, expected sm_<major><minor>"
    with pytest.raises(ValueError, match=err):
        ct.ByTarget(sm100=4)


def _dummy():
    pass


@pytest.mark.parametrize("value", [None, 4, 8])
def test_num_worker_warps_accepts_valid(value):
    ct.kernel(_dummy, num_worker_warps=value)


@pytest.mark.parametrize("value", [3, 7, 10])
def test_num_worker_warps_rejects_invalid(value):
    with pytest.raises(ValueError, match="num_worker_warps should be either 4 or 8"):
        ct.kernel(_dummy, num_worker_warps=value)


def test_num_worker_warps_accepts_by_target():
    ct.kernel(_dummy, num_worker_warps=ct.ByTarget(sm_100=8, default=4))
