# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0


def set_bits_fixed(value, field, position, width, full_mask):
    field_mask = (1 << width) - 1
    mask = field_mask << position
    clear_mask = full_mask - mask
    insert = (field & field_mask) << position
    return (value & clear_mask) | insert


def set_bits32(value, field, position, width):
    return set_bits_fixed(value, field, position, width, 0xFFFF_FFFF)


def set_bits64(value, field, position, width):
    return set_bits_fixed(value, field, position, width, 0xFFFF_FFFF_FFFF_FFFF)


def set_bit32(value, position, bit=1):
    return set_bits32(value, bit, position, 1)


def set_bit64(value, position, bit=1):
    return set_bits64(value, bit, position, 1)
