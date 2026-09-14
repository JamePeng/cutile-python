# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum, IntEnum

import pytest
import torch

import cuda.tile as ct
from cuda.tile import TileTypeError, TileValueError
from cuda.tile._ir.op_impl import (
    require_constant_enum,
    require_optional_constant_enum,
)
from cuda.tile._ir.type import StringTy


class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2


class Status(Enum):
    OK = "ok"
    ERROR = "error"


class Weight(Enum):
    LIGHT = 0.5
    HEAVY = 2.0


class Priority(IntEnum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2


class Spelling(Enum):
    UPPER = 0
    lower = 1
    MixedCase = 2
    has_underscore = 3
    ALIAS = 0


class FakeIRConst:
    def __init__(self, value):
        self.value = value

    def is_constant(self):
        return True

    def get_constant(self):
        return self.value

    def get_type(self):
        return StringTy(self.value)


@pytest.mark.parametrize("name,expected", Spelling.__members__.items())
def test_enum_requirements_accept_str(name, expected):
    assert require_constant_enum(FakeIRConst(name), Spelling) is expected
    assert require_optional_constant_enum(FakeIRConst(name), Spelling) is expected


@pytest.mark.parametrize("name", ('upper', 'LOWER'))
def test_require_constant_enum_rejects_wrong_case(name):
    with pytest.raises(TileTypeError, match="Expected Spelling"):
        require_constant_enum(FakeIRConst(name), Spelling)


def test_comparison_eq():
    @ct.kernel
    def kernel(out):
        x = Color.RED
        if x == Color.RED:
            ct.scatter(out, (), 1)
        else:
            ct.scatter(out, (), -1)

    out = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (out,))
    assert out.item() == 1


def test_comparison_not_equal():
    @ct.kernel
    def kernel(out):
        x = Color.RED
        if x != Color.GREEN:
            ct.scatter(out, (), 1)
        else:
            ct.scatter(out, (), -1)

    out = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (out,))
    assert out.item() == 1


def test_construction_from_known_int():
    @ct.kernel
    def kernel(out):
        i = 0
        x = Color(i)
        if x == Color.RED:
            ct.scatter(out, (), 10)
        elif x == Color.GREEN:
            ct.scatter(out, (), 20)
        else:
            ct.scatter(out, (), 30)

    out = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (out,))
    assert out.item() == 10


def test_construction_from_string_value():
    @ct.kernel
    def kernel(out):
        x = Status("ok")
        if x == Status.OK:
            ct.scatter(out, (), 1)
        else:
            ct.scatter(out, (), 0)

    out = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (out,))
    assert out.item() == 1


def test_construction_from_float_value():
    @ct.kernel
    def kernel(out):
        x = Weight(0.5)
        if x == Weight.LIGHT:
            ct.scatter(out, (), 1)
        else:
            ct.scatter(out, (), 0)

    out = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (out,))
    assert out.item() == 1


def test_intenum_ordering():
    @ct.kernel
    def kernel(out):
        if Priority.LOW < Priority.HIGH:
            ct.scatter(out, (), 1)
        else:
            ct.scatter(out, (), -1)

    out = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (out,))
    assert out.item() == 1


# ===========================================================================
# Error cases
# ===========================================================================

def test_construction_from_runtime_value_raises():
    @ct.kernel
    def kernel(x, out):
        bid = ct.bid(0)
        _ = Color(bid)
        ct.scatter(out, (), 0)

    x = torch.zeros(1, device="cuda:0")
    out = torch.zeros((), dtype=torch.int32, device="cuda:0")
    with pytest.raises(TileTypeError):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (x, out))


@pytest.mark.parametrize("invalid_value", ["foo", 99])
def test_construction_from_invalid_type_or_value_raises(invalid_value):
    @ct.kernel
    def kernel(out):
        _ = Color(invalid_value)
        ct.scatter(out, (), 0)

    out = torch.zeros((), dtype=torch.int32, device="cuda:0")
    with pytest.raises(TileValueError):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (out,))


@pytest.mark.parametrize("enum_value", [Color.BLUE, Status.ERROR,  Weight.HEAVY])
def test_name_attribute(enum_value):
    name = enum_value.name

    @ct.kernel
    def kernel(out):
        if enum_value.name == name:
            ct.scatter(out, (), 1)

    out = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (out,))
    assert out.item() == 1


@pytest.mark.parametrize("enum_value", [Color.BLUE, Status.ERROR,  Weight.HEAVY])
def test_value_attribute(enum_value):
    value = enum_value.value

    @ct.kernel
    def kernel(out):
        if enum_value.value == value:
            ct.scatter(out, (), 1)

    out = torch.zeros((), dtype=torch.int32, device="cuda:0")
    ct.launch(torch.cuda.current_stream(), (1,), kernel, (out,))
    assert out.item() == 1


def test_enum_ordering_raises():
    @ct.kernel
    def kernel(out):
        if Color.RED < Color.GREEN:
            ct.scatter(out, (), 1)
        else:
            ct.scatter(out, (), -1)

    out = torch.zeros((), dtype=torch.int32, device="cuda:0")
    with pytest.raises(TileTypeError):
        ct.launch(torch.cuda.current_stream(), (1,), kernel, (out,))
