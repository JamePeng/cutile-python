# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the task-scheduled clustered FP16/BF16 GEMM tutorial."""

import importlib

import pytest
import torch

from task_scheduling_test_requirements import task_scheduling as ts
from task_scheduling_test_utils import require_blackwell_cc100


_MODULE = (
    "experimental.task_scheduling.tutorial.04_gemm_bf16_advanced."
    "01_fp16_bf16_gemm_3_cluster"
)


def require_available_blackwell():
    if not torch.cuda.is_available():
        return pytest.mark.skip(reason="requires a Blackwell CC 10.x GPU")
    return require_blackwell_cc100()


@pytest.mark.parametrize("clc_dynamic_scheduler", [False, True])
def test_gemm_advanced_functional_mma_loop_capture(clc_dynamic_scheduler):
    """Keep the carried MMA body in the captured work-tile schedule."""
    kernel = importlib.import_module(_MODULE)
    pipeline = kernel._create_gemm_pipeline(
        512,
        512,
        use_clc_dynamic_scheduler=clc_dynamic_scheduler,
    )
    work_tile_loop = pipeline.mma_task.schedule.body[3]
    domain_loop = next(
        node for node in work_tile_loop.body if isinstance(node, ts.DomainLoop)
    )

    assert tuple(domain_loop.initial_values) == ("scale_d",)
    assert tuple(domain_loop.yield_values) == ("scale_d",)
    assert tuple(domain_loop.result_values) == ("scale_d",)
    assert [step.label or step.schedule_stage.value for step in domain_loop.body] == [
        "ConsumerTryWait",
        "ConsumerTryWait",
        "ConsumerWait",
        "ConsumerWait",
        "build_desc_a",
        "build_desc_b",
        "mma",
        "ConsumerRelease",
        "ConsumerRelease",
    ]


@pytest.mark.parametrize("clc_dynamic_scheduler", [False, True])
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@pytest.mark.parametrize(
    "mnk",
    [
        (512, 256, 256),
        (256, 512, 2048),
    ],
)
@require_available_blackwell()
def test_fp16_bf16_gemm_3_prim_ts_cluster(
    mnk, has_bias, clc_dynamic_scheduler, dtype
):
    """Run static and CLC clustered GEMMs on tile-aligned shapes."""
    kernel = importlib.import_module(_MODULE)

    if dtype == "bf16":
        tolerance = 2.0e-2
    else:
        # The CUDA Lang FP16 bias epilogue can differ from the FP32 reference
        # by one FP16 ULP after the final conversion.
        tolerance = 1.0e-3 if has_bias else 1.0e-4
    kernel.verify(
        mnk,
        tolerance=tolerance,
        has_bias=has_bias,
        dtype=dtype,
        use_clc_dynamic_scheduler=clc_dynamic_scheduler,
    )


@pytest.mark.parametrize("dtype", ["fp16", "bf16"])
@require_available_blackwell()
def test_fp16_bf16_gemm_3_prim_ts_cluster_bias_non_aligned_n(dtype):
    """Verify bias broadcasting for a non-tile-aligned output."""
    kernel = importlib.import_module(_MODULE)

    tolerance = 1.0e-4 if dtype == "fp16" else 2.0e-2
    kernel.verify(
        (32, 10, 64),
        tolerance=tolerance,
        has_bias=True,
        dtype=dtype,
        use_clc_dynamic_scheduler=False,
    )
