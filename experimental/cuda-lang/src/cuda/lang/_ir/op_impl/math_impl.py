# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cuda.lang._datatype as datatype
from cuda.lang._exception import TileTypeError
from cuda.lang._ir.ir import Var, add_operation
from cuda.lang._ir.type import (
    ScalarTy,
    VectorTy,
)
from cuda.lang._ir.op_defs import RawNVVMIntrinsic, RawMLIROperation
from cuda.lang._ir.type_checking_helpers import (
    common_type,
    require_scalar_or_vector_float_type,
    require_scalar_or_vector_type,
)
from cuda.tile._datatype import is_float, is_integral
from cuda.tile._ir.arithmetic_ops import astype, promote_and_broadcast_to
from cuda.tile._ir.core_ops import strictly_typed_const
from cuda.tile._ir.op_impl import (
    ImplRegistry,
)
from ..._stub import math as cl_math


_registry = ImplRegistry()
impl = _registry.impl


def math_impl_registry() -> ImplRegistry:
    return _registry


@impl(cl_math.ceil, fixed_args=["math.ceil"])
@impl(cl_math.exp, fixed_args=["math.exp"])
@impl(cl_math.exp2, fixed_args=["math.exp2"])
@impl(cl_math.sin, fixed_args=["math.sin"])
@impl(cl_math.cos, fixed_args=["math.cos"])
@impl(cl_math.tan, fixed_args=["math.tan"])
@impl(cl_math.sinh, fixed_args=["math.sinh"])
@impl(cl_math.cosh, fixed_args=["math.cosh"])
@impl(cl_math.tanh, fixed_args=["math.tanh"])
@impl(cl_math.sqrt, fixed_args=["math.sqrt"])
@impl(cl_math.rsqrt, fixed_args=["math.rsqrt"])
@impl(cl_math.floor, fixed_args=["math.floor"])
@impl(cl_math.log, fixed_args=["math.log"])
@impl(cl_math.log2, fixed_args=["math.log2"])
def math_float_unary_impl(op_name: str, x: Var):
    x_ty = require_scalar_or_vector_float_type(x)
    return add_operation(
        RawMLIROperation,
        x_ty,
        op_name=op_name,
        operands_=(x,),
    )


@impl(cl_math.isnormal)
def math_isnormal_impl(x: Var):
    x_ty = require_scalar_or_vector_type(x, datatype.is_unrestricted_float)
    match x_ty:
        case ScalarTy():
            res_ty = ScalarTy(datatype.bool_)
        case VectorTy(length=length):
            res_ty = VectorTy(datatype.bool_, length=length)
        case _:
            assert False
    # see https://llvm.org/docs/LangRef.html#llvm-is-fpclass-intrinsic
    mask = strictly_typed_const((1 << 3) | (1 << 8), ScalarTy(datatype.int32))
    return add_operation(
        RawNVVMIntrinsic,
        res_ty,
        intrinsic="llvm.is.fpclass",
        operands_=(x, mask),
    )


@impl(cl_math.isnan, fixed_args=["math.isnan"])
@impl(cl_math.isinf, fixed_args=["math.isinf"])
@impl(cl_math.isfinite, fixed_args=["math.isfinite"])
def math_float_fpclass_impl(op_name: str, x: Var):
    x_ty = require_scalar_or_vector_type(x, datatype.is_float)
    match x_ty:
        case ScalarTy():
            res_ty = ScalarTy(datatype.bool_)
        case VectorTy(length=length):
            res_ty = VectorTy(datatype.bool_, length=length)
        case _:
            assert False
    return add_operation(
        RawMLIROperation,
        res_ty,
        op_name=op_name,
        operands_=(x,),
    )


@impl(cl_math.pow)
def math_pow_impl(x: Var, y: Var):
    x_ty, y_ty = x.get_type(), y.get_type()
    base_dt, exp_dt = x_ty.tensor_dtype(), y_ty.tensor_dtype()

    def smallest_float_dtype(dtype):
        if is_float(dtype):
            return dtype
        if is_integral(dtype):
            return datatype.float32 if dtype.bitwidth <= 32 else datatype.float64
        raise TileTypeError(f"math.pow expects arithmetic operands, got {dtype}")

    x = astype(x, smallest_float_dtype(base_dt))
    y = astype(y, smallest_float_dtype(exp_dt))
    common_ty = common_type(x, y)
    x = promote_and_broadcast_to(x, common_ty)
    y = promote_and_broadcast_to(y, common_ty)
    result_ty = common_ty
    cast_down_to_float16 = result_ty.tensor_dtype() == datatype.float16 or (
        base_dt == datatype.float16 and is_integral(exp_dt)
    )
    if cast_down_to_float16:
        # NOTE: powi with a f16 base fails in ISEL; convert to f32 and back
        x = astype(x, datatype.float32)
        y = astype(y, datatype.float32)
        result_ty = x.get_type()
    result = add_operation(
        RawMLIROperation,
        result_ty,
        op_name="math.powf",
        operands_=(x, y),
    )
    if cast_down_to_float16:
        return astype(result, datatype.float16)
    return result


@impl(cl_math.atan2, fixed_args=["math.atan2"])
def math_float_binary_impl(op_name: str, x: Var, y: Var):
    require_scalar_or_vector_float_type(x)
    require_scalar_or_vector_float_type(y)
    ty = common_type(x, y)
    x = promote_and_broadcast_to(x, ty)
    y = promote_and_broadcast_to(y, ty)
    return add_operation(
        RawMLIROperation,
        ty,
        op_name=op_name,
        operands_=(x, y),
    )


@impl(cl_math.abs)
def abs_impl(x: Var) -> Var:
    x_ty = require_scalar_or_vector_type(x)
    x_dtype = x_ty.tensor_dtype()
    if datatype.is_float(x_dtype):
        op_name = "math.absf"
    elif datatype.is_integral(x_dtype):
        # If it's unsigned, then the absolute value is the identity
        if not datatype.is_signed(x_dtype):
            return x
        op_name = "math.absi"
    else:
        raise TileTypeError(f"abs() expects an arithmetic scalar, got {x_ty}")
    return add_operation(
        RawMLIROperation,
        x_ty,
        op_name=op_name,
        operands_=(x,),
    )
