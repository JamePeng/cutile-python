# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal device helpers used by task-scheduling lowering."""

import cuda.lang as cl


@cl.function()
def block_in_cluster_rank():
    """Read the current CTA's linear rank without extending the public API."""
    return cl._nvvm.read_ptx_sreg_cluster_ctarank()
