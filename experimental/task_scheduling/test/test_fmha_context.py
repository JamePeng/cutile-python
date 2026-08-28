# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SM120 task-scheduled FMHA context kernel."""

import importlib

import pytest
import torch

from task_scheduling_test_utils import require_blackwell_cc120


_FMHA_PACKAGE = "experimental.task_scheduling.tutorial.06_fmha_context"


def import_fmha_module(name):
    return importlib.import_module(f"{_FMHA_PACKAGE}.{name}")


def require_available_sm120():
    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="requires a Blackwell CC 12.0 GPU")
    return require_blackwell_cc120()


# -------------------------------------------------------------------------
# Schedule-only validation (no GPU required)
# -------------------------------------------------------------------------

_SCHEDULE_CONFIGS = [
    {"is_causal": True, "head_dim": 128},
    {"is_causal": False, "head_dim": 128},
    {"is_causal": True, "head_dim": 64},
]


@pytest.mark.parametrize(
    "cfg",
    _SCHEDULE_CONFIGS,
    ids=["causal_d128", "non_causal_d128", "causal_d64"],
)
def test_fmha_ts_schedule_validation(cfg):
    """Validate the SM120 FMHA schedule without GPU execution."""
    run_module = import_fmha_module("fmha_run")
    head_dim = cfg["head_dim"]
    run_module.validate_schedule(
        q_shape=(2, 1024, 8, head_dim),
        k_shape=(2, 1024, 8, head_dim),
        is_causal=cfg["is_causal"],
    )


# -------------------------------------------------------------------------
# GPU accuracy tests (sm_12x only)
# -------------------------------------------------------------------------

_ACCURACY_CONFIGS = [
    # (q_shape, k_shape, is_causal)
    pytest.param((2, 512, 8, 128), (2, 512, 8, 128), False),
    pytest.param((2, 512, 8, 128), (2, 512, 8, 128), True),
    pytest.param((2, 512, 8, 64), (2, 512, 8, 64), True),
    pytest.param((1, 1024, 8, 128), (1, 1024, 8, 128), False),
]


@pytest.mark.parametrize(
    ("q_shape", "k_shape", "is_causal"),
    _ACCURACY_CONFIGS,
    ids=[
        "b2_s512_d128",
        "b2_s512_d128_causal",
        "b2_s512_d64_causal",
        "b1_s1024_d128",
    ],
)
@require_available_sm120()
@pytest.mark.skip(reason="https://jirasw.nvidia.com/browse/CFK-38236")
def test_fmha_ts_accuracy(q_shape, k_shape, is_causal):
    """Run the SM120 FMHA kernel and compare it with the PyTorch reference."""
    resources_module = import_fmha_module("fmha_resources")
    run_module = import_fmha_module("fmha_run")
    cfg = resources_module.FmhaConfig(
        head_dim=q_shape[3],
        is_causal=is_causal,
    )
    tensors = run_module.prepare_tensors(q_shape, k_shape)
    run_module.run(tensors, cfg)
    torch.cuda.synchronize()
    run_module.verify_output(tensors, cfg, tolerance=0.1)
