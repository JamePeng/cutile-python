# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from ._nvvm_support import IX, B


def get_bits(value: IX, position: IX, width: IX) -> IX:
    mask = (1 << width) - 1
    return (value >> position) & mask


def set_bits(value: IX, field: IX, position: IX, width: IX) -> IX:
    field_mask = (1 << width) - 1
    mask = field_mask << position
    return (value & ~mask) | ((field & field_mask) << position)


def get_bit(value: IX, position: IX) -> B:
    return get_bits(value, position, 1)


def set_bit(value: IX, position: IX, bit: B = 1) -> IX:
    return set_bits(value, bit, position, 1)
