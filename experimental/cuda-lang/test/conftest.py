# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

pytest.importorskip("cuda.lang", reason="Skipping cuda-lang test: module not found")


@pytest.fixture(autouse=True)
def reset_torch_device():
    torch.cuda.set_device(0)
