# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Literal

from cuda.lang._execution import stub
from .bits import set_bit, set_bits
from .nvvm import P3, P6


class CTAGroup(Enum):
    """CTA group selection for tcgen05 tensor memory operations."""

    CTA_1 = "cg1"
    CTA_2 = "cg2"


class Tcgen05LdStShape(Enum):
    """Load/store shapes supported by tcgen05 tensor memory operations."""

    SHAPE_16X64B = "16x64b"
    SHAPE_16X128B = "16x128b"
    SHAPE_16X256B = "16x256b"
    SHAPE_32X32B = "32x32b"
    SHAPE_16X32BX2 = "16x32bx2"


@stub
def tcgen05_alloc(
    addr: P3,
    ncols: int,
    *,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None:
    """Allocate tensor memory columns and write the tensor-memory address to ``addr``."""
    ...


@stub
def tcgen05_dealloc(
    addr: P6,
    ncols: int,
    *,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None:
    """Deallocate tensor memory columns starting at ``addr``."""
    ...


@stub
def tcgen05_commit(
    mbar: P3,
    *,
    multicast_mask: int | None = None,
    cta_group: CTAGroup = CTAGroup.CTA_1,
) -> None:
    """Commit tcgen05 tensor memory operations and arrive at ``mbar``."""
    ...


@stub
def tcgen05_ld(
    shape: Tcgen05LdStShape,
    tmem_addr: P6,
    *,
    count: int = 1,
    pack: bool | None = None,
    offset: int | None = None,
) -> Any:
    """Load registers from tensor memory using a tcgen05 load shape."""
    ...


class _Tcgen05Tf32Type(IntEnum):
    TF32 = 2


class _Tcgen05F16Type(IntEnum):
    F16 = 0
    BF16 = 1


class _Tcgen05F8F6F4Type(IntEnum):
    E4M3 = 0
    E5M2 = 1
    E2M3 = 3
    E3M2 = 4
    E2M1 = 5


class _Tcgen05I8Type(IntEnum):
    U8 = 0
    S8 = 1


class _Tcgen05Mxf4Type(IntEnum):
    E2M1 = 1


class _DType(IntEnum):
    F16 = 0
    F32 = 1
    S32 = 2


class _MaxShift(IntEnum):
    NoShift = 0
    MaxShift8 = 1
    MaxShift16 = 2
    MaxShift32 = 3


class _Mxf8f6f4ScaleFormat(IntEnum):
    UE8M0 = 1


class _Mxf4ScaleFormat(IntEnum):
    UE4M3 = 0
    UE8M0 = 1


class _Mxf4KDimension(IntEnum):
    DenseK64OrSparseK128 = 0
    DenseK96 = 1


@dataclass(frozen=True)
class Tcgen05InstructionDescriptor:
    """
    Instruction descriptor format for .kind::tf32, .kind::f16, .kind::f8f6f4 and .kind::i8
    """

    Tf32Type = _Tcgen05Tf32Type
    F16Type = _Tcgen05F16Type
    F8F6F4Type = _Tcgen05F8F6F4Type
    I8Type = _Tcgen05I8Type
    DType = _DType
    MaxShift = _MaxShift

    sparsity_selector: int = 0
    sparse: bool = False
    saturate: bool = False
    d_type: DType = DType.F16
    a_type: Tf32Type | F16Type | F8F6F4Type | I8Type = F16Type.F16
    b_type: Tf32Type | F16Type | F8F6F4Type | I8Type = F16Type.F16
    negate_a: bool = False
    negate_b: bool = False
    transpose_a: bool = False
    transpose_b: bool = False
    n: int = 0
    m: int = 0
    max_shift: MaxShift = MaxShift.NoShift

    def encode(self) -> int:
        desc = 0
        desc = set_bits(desc, self.sparsity_selector, 0, 2)
        desc = set_bit(desc, 2, self.sparse)
        desc = set_bit(desc, 3, self.saturate)
        desc = set_bits(desc, self.d_type, 4, 2)
        desc = set_bits(desc, self.a_type, 7, 3)
        desc = set_bits(desc, self.b_type, 10, 3)
        desc = set_bit(desc, 13, self.negate_a)
        desc = set_bit(desc, 14, self.negate_b)
        desc = set_bit(desc, 15, self.transpose_a)
        desc = set_bit(desc, 16, self.transpose_b)
        desc = set_bits(desc, self.n >> 3, 17, 6)
        desc = set_bits(desc, self.m >> 4, 24, 5)
        desc = set_bits(desc, self.max_shift, 30, 2)
        return desc


@dataclass(frozen=True)
class Tcgen05Mxf8f6f4InstructionDescriptor:
    """Instruction descriptor format for .kind::mxf8f6f4"""

    Type = _Tcgen05F8F6F4Type
    ScaleFormat = _Mxf8f6f4ScaleFormat

    sparse: bool = False
    b_scale_id: Literal[0, 1, 2, 3] = 0
    a_type: Type = Type.E4M3
    b_type: Type = Type.E4M3
    negate_a: bool = False
    negate_b: bool = False
    transpose_a: bool = False
    transpose_b: bool = False
    n: int = 0
    scale_format: ScaleFormat = ScaleFormat.UE8M0
    m: int = 0
    a_scale_id: Literal[0, 1, 2, 3] = 0

    def encode(self) -> int:
        desc = 0
        desc = set_bit(desc, 2, self.sparse)
        desc = set_bits(desc, self.b_scale_id, 4, 2)
        desc = set_bits(desc, self.a_type, 7, 3)
        desc = set_bits(desc, self.b_type, 10, 3)
        desc = set_bit(desc, 13, self.negate_a)
        desc = set_bit(desc, 14, self.negate_b)
        desc = set_bit(desc, 15, self.transpose_a)
        desc = set_bit(desc, 16, self.transpose_b)
        desc = set_bits(desc, self.n >> 3, 17, 6)
        desc = set_bit(desc, 23, self.scale_format)
        desc = set_bits(desc, self.m >> 7, 27, 2)
        desc = set_bits(desc, self.a_scale_id, 29, 2)
        return desc


@dataclass(frozen=True)
class Tcgen05Mxf4InstructionDescriptor:
    """Instruction descriptor format for .kind::mxf4 and .kind::mxf4nvf4"""

    Type = _Tcgen05Mxf4Type
    ScaleFormat = _Mxf4ScaleFormat
    KDimension = _Mxf4KDimension

    sparse: bool = False
    b_scale_id: Literal[0, 2] = 0
    a_type: Type = Type.E2M1
    b_type: Type = Type.E2M1
    negate_a: bool = False
    negate_b: bool = False
    transpose_a: bool = False
    transpose_b: bool = False
    n: int = 0
    scale_format: ScaleFormat = ScaleFormat.UE8M0
    m: int = 0
    a_scale_id: Literal[0, 2] = 0
    k_dimension: KDimension = KDimension.DenseK64OrSparseK128

    def encode(self) -> int:
        desc = 0
        desc = set_bit(desc, 2, self.sparse)
        desc = set_bits(desc, self.b_scale_id, 4, 2)
        desc = set_bits(desc, self.a_type, 7, 3)
        desc = set_bits(desc, self.b_type, 10, 2)
        desc = set_bit(desc, 13, self.negate_a)
        desc = set_bit(desc, 14, self.negate_b)
        desc = set_bit(desc, 15, self.transpose_a)
        desc = set_bit(desc, 16, self.transpose_b)
        desc = set_bits(desc, self.n >> 3, 17, 6)
        desc = set_bit(desc, 23, self.scale_format)
        desc = set_bits(desc, self.m >> 7, 27, 2)
        desc = set_bits(desc, self.a_scale_id, 29, 2)
        desc = set_bit(desc, 31, self.k_dimension)
        return desc


__all__ = (
    "CTAGroup",
    "Tcgen05LdStShape",
    "Tcgen05InstructionDescriptor",
    "Tcgen05Mxf8f6f4InstructionDescriptor",
    "Tcgen05Mxf4InstructionDescriptor",
    "tcgen05_alloc",
    "tcgen05_dealloc",
    "tcgen05_commit",
    "tcgen05_ld",
)
