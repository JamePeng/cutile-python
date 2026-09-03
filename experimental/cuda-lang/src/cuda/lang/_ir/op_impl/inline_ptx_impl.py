# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Sequence

from cuda.tile._datatype import is_pointer_dtype, is_integral
from cuda.tile._exception import InvalidValueError
from cuda.tile._ir.arithmetic_ops import astype
from cuda.tile._ir.op_impl import (
    require_constant_str,
    ImplRegistry, make_type_checking_error,
)
from cuda.tile._ir.core_ops import (
    build_tuple,
)
from cuda.tile._ir.ir import add_operation_variadic
from cuda.lang._exception import TypeCheckingError
import cuda.lang._datatype as datatype
from cuda.tile._ir.type import DTypeSpec, TensorLikeTy
from ..op_defs import InlinePTX, InlineAsmPiece, InlineAsmOutput, \
    InlineAsmInput
from ..type import (
    ScalarTy,
    PointerTy,
    TupleTy,
)
from ..ir import Var
from ..._stub import core_api

_registry = ImplRegistry()
impl = _registry.impl


def inline_ptx_impl_registry() -> ImplRegistry:
    return _registry


@impl(core_api._inline_ptx)
def inline_ptx_impl(ptx_code: Var, args: tuple[Var]) -> Var[TupleTy]:
    ptx_code = require_constant_str(ptx_code)
    args = [_require_arg(x) for x in args]
    results = inline_ptx(ptx_code, *args)
    return build_tuple(results)


def _require_arg(arg: Var) -> Var[TensorLikeTy] | datatype.DType:
    ty = arg.get_type()
    if isinstance(ty, DTypeSpec):
        return ty.dtype

    if not isinstance(ty, ScalarTy | PointerTy):
        raise make_type_checking_error(
            f"Inline PTX inputs must be scalars or pointers,"
            f" but given value has type {ty}", arg)
    return arg


SUPPORTED_BITWIDTHS = (16, 32, 64, 128)


def inline_ptx(ptx_code: str, *args: Var | datatype.DType) -> tuple[Var, ...]:
    inputs = []
    result_types = []
    arg_pieces = []
    result_converters = []
    for arg in args:
        if isinstance(arg, datatype.DType):
            arg_pieces.append(InlineAsmOutput(len(result_types)))
            ptx_dtype, _, from_ptx = _to_supported_dtype(arg)
            result_types.append(PointerTy(ptx_dtype)
                                if is_pointer_dtype(ptx_dtype) else ScalarTy(ptx_dtype))
            result_converters.append(from_ptx)
        else:
            arg_pieces.append(InlineAsmInput(len(inputs)))
            dtype = arg.get_type().tensor_dtype()
            ptx_dtype, to_ptx, _ = _to_supported_dtype(dtype)
            arg = to_ptx(arg)
            inputs.append(arg)

    text = _parse_inline_asm_text(ptx_code, arg_pieces)
    results = add_operation_variadic(
        InlinePTX,
        tuple(result_types),
        text=text,
        inputs=tuple(inputs),
    )
    return tuple(conv(x) for x, conv in zip(results, result_converters, strict=True))


def _to_supported_dtype(dtype: datatype.DType):
    if dtype.bitwidth not in (8, 16, 32, 64, 128):
        raise TypeCheckingError(f"{dtype.bitwidth}-bit arguments are not supported")

    if dtype.bitwidth == 8 and dtype != datatype.bool_:
        from .core_api_impl import bitcast
        target_dtype = datatype.integer_dtype(dtype.bitwidth, signed=datatype.is_signed(dtype))

        if is_integral(dtype):
            int_dtype = dtype
        else:
            int_dtype = datatype.integer_dtype(dtype.bitwidth, signed=False)

        def to_ptx(x: Var):
            if dtype != int_dtype:
                x = bitcast(x, int_dtype)
            return astype(x, target_dtype)

        def from_ptx(x: Var):
            x = astype(x, int_dtype)
            if dtype != int_dtype:
                x = bitcast(x, dtype)
            return x

        return target_dtype, to_ptx, from_ptx

    return dtype, lambda x: x, lambda x: x


# Match either a placeholder (e.g. %123), an escaped percent sign (%%), a stray %, or the end.
_PLACEHOLDER_REGEX = re.compile("%([0-9]+|%?)|$")


def _parse_inline_asm_text(text: str,
                           args: Sequence[InlineAsmInput | InlineAsmOutput]
                           ) -> tuple[InlineAsmPiece]:
    start = 0
    pieces = []
    while True:
        m = _PLACEHOLDER_REGEX.search(text, start)
        pieces.append(text[start:m.start()])
        match m[1]:
            case None: break  # Reached the end of the string
            case "%": pieces.append("%")  # '%%' escape
            case "": raise InvalidValueError(
                "Literal percent signs in inline PTX must be escaped as '%%'")
            case index:
                index = int(index)
                if index >= len(args):
                    raise TypeCheckingError(f"inline_ptx placeholder %{index} is out of range "
                                            f"for {len(args)} arguments")
                pieces.append(args[index])
        start = m.end()
    return tuple(pieces)
