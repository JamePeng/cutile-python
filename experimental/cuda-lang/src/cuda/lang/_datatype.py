# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import TypeAlias, Union

from cuda.lang._ir.type import (
    MemorySpace,
    TileTy,
)
from cuda.tile._stub import Tile
from cuda.tile._datatype import (
    DType,
    bfloat16,
    bool_,
    float8_e8m0fnu,
    float4_e2m1fn,
    float8_e4m3fn,
    float8_e5m2,
    float16,
    float32,
    float64,
    int8,
    int16,
    int32,
    int64,
    tfloat32,
    uint8,
    uint16,
    uint32,
    uint64,
    is_float,
    is_restricted_float,
    is_unrestricted_float,
    is_arithmetic,
    is_boolean,
    is_integral,
    is_signed,
    get_signedness,
    default_int_type,
    integer_dtype,
    is_pointer_dtype,
    pointer_dtype,
    opaque_pointer_dtype,
    _define_dtype, _DTypeDefinition,
)


mbarrier = _define_dtype('mbarrier', _DTypeDefinition(bitwidth=64))
clusterlaunchcontrol_token = _define_dtype(
    "clusterlaunchcontrol_token", _DTypeDefinition(bitwidth=128)
)


def to_torch_dtype(dtype: DType | TileTy):
    import torch

    if isinstance(dtype, TileTy):
        dtype = dtype.dtype

    dtype_map = {
        bool_: torch.bool,
        uint8: torch.uint8,
        uint16: torch.uint16,
        uint32: torch.uint32,
        uint64: torch.uint64,
        int8: torch.int8,
        int16: torch.int16,
        int32: torch.int32,
        int64: torch.int64,
        float16: torch.float16,
        bfloat16: torch.bfloat16,
        float32: torch.float32,
        float64: torch.float64,
    }
    if dtype in dtype_map:
        return dtype_map[dtype]

    optional_dtype_names = {
        float8_e4m3fn: "float8_e4m3fn",
        float8_e5m2: "float8_e5m2",
        float8_e8m0fnu: "float8_e8m0fnu",
    }
    if dtype in optional_dtype_names and hasattr(torch, optional_dtype_names[dtype]):
        return getattr(torch, optional_dtype_names[dtype])

    raise NotImplementedError(f"No torch dtype mapping for {dtype}")


TypeSpec: TypeAlias = Union[DType | TileTy]


def is_any_pointer(value):
    return isinstance(value, Tile) and value.ndim == 0 and is_pointer_dtype(value.dtype)


__all__ = [
    "is_float",
    "is_restricted_float",
    "is_unrestricted_float",
    "is_arithmetic",
    "is_boolean",
    "is_integral",
    "is_signed",
    "is_any_pointer",
    "is_pointer_dtype",
    "pointer_dtype",
    "opaque_pointer_dtype",
    "get_signedness",
    "integer_dtype",
    "bool_",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    "bfloat16",
    "tfloat32",
    "float8_e4m3fn",
    "float8_e5m2",
    "float8_e8m0fnu",
    "float4_e2m1fn",
    "mbarrier",
    "clusterlaunchcontrol_token",
    "DType",
    "to_torch_dtype",
    "default_int_type",
    "MemorySpace",
]
