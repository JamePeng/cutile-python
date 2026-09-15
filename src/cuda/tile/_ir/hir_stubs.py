# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from cuda.tile._execution import stub


if TYPE_CHECKING:
    from cuda.tile._ir.hir import StaticEvalExpression, Function


@stub(compiled_host=True)
def if_else(cond, then_block, else_block, /): ...


@stub(compiled_host=True)
def tuple_comp_if(cond, then_block, /): ...  # Static-only if: rejects dynamic conditions


@stub(compiled_host=True)
def loop(body, iterable, /): ...  # infinite if `iterable` is None


@stub(compiled_host=True)
def static_foreach(body, items, /): ...


@stub(compiled_host=True)
def build_tuple(*items): ...  # Makes a tuple (i.e. returns `items`)


@stub(compiled_host=True)
def build_formatted_string(format, *values): ...  # Creates a FormattedStringTy value


@stub(compiled_host=True)
def unpack(iterable, expected_len, /): ...


@stub(compiled_host=True)
def identity(x): ...   # Identity function (i.e. returns `x`)


@stub(compiled_host=True)
def store_var(rn, value, /): ...  # Store value into the local slot given by ResolvedName


@stub(compiled_host=True)
def load_var(rn, name, /): ...  # Load from the slot/global given by ResolvedName


@stub(compiled_host=True)
def make_closure(func_hir: "Function", /, *default_values): ...


@stub(compiled_host=True)
def do_static_eval(expr: "StaticEvalExpression", *local_var_values): ...


@stub(compiled_host=True)
def do_static_assert(condition, message_block, /): ...


@stub(compiled_host=True)
def enter_context(manager, /): ...


@stub(compiled_host=True)
def pop_context(): ...


@stub(compiled_host=True)
def is_contained_in(x, y, /): ...  # "return x in y"


@stub(compiled_host=True)
def is_not_contained_in(x, y, /): ...  # return "x not in y"


@stub
def generator_yield(value, /): ...
