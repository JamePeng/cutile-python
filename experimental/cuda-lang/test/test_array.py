# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Annotated

import cuda.lang as cl
import torch


def test_array_constant_shape_strides():
    @cl.kernel
    def kern(arr: Annotated[cl.Array, cl.ArrayAnnotation(static_shape_dims=(0,),
                                                         static_stride_dims=(1,))]):
        shape = arr.shape
        strides = arr.strides
        cl.static_assert(shape[0] == 5)
        cl.static_assert(strides[1] == 1)
        cl.static_assert(isinstance(shape[1], cl.Scalar))
        cl.static_assert(isinstance(strides[0], cl.Scalar))

    cl.launch(torch.cuda.current_stream(), (1,), (1,), kern, (torch.ones((5, 7), device="cuda"),))
