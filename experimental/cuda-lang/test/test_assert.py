# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang as cl

import argparse
import sys
import subprocess
import torch
import pytest


def do_test_device_assertions(condition, has_message):
    @cl.kernel
    def kernel(tensor):
        cl.assert_(tensor[0], message="failure message" if has_message else None)

    tensor = torch.tensor([condition], dtype=torch.int32).cuda()
    cl.launch(torch.cuda.current_stream(), (1,), (1,), kernel, (tensor,))
    torch.cuda.synchronize()


@pytest.mark.parametrize("condition", [False, True], ids=lambda x: f"condition={x}")
@pytest.mark.parametrize("has_message", [False, True], ids=lambda x: f"has_message={x}")
def test_device_assertions(condition, has_message):
    args = [sys.executable, __file__]
    if condition:
        args.append("--condition")
    if has_message:
        args.append("--has-message")
    proc = subprocess.run(args, capture_output=True)
    if condition:
        assert proc.returncode == 0
    else:
        assert proc.returncode != 0
        if has_message:
            actual_outs = [
                line for line in proc.stdout.decode("UTF-8").splitlines() if line
            ]
            assert len(actual_outs) == 1
            assert "failure message" in actual_outs[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="store_true")
    parser.add_argument("--has-message", action="store_true")
    args = parser.parse_args()
    do_test_device_assertions(args.condition, args.has_message)
