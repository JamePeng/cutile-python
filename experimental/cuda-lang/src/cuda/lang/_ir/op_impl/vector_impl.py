# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import operator
from typing import Callable
from cuda.tile._ir.op_impl import (
    WILDCARD,
    ImplRegistry,
)
from cuda.tile._ir.cast_ops import implicit_cast
from cuda.tile._ir.ops import strictly_typed_const
from cuda.lang._exception import TileInternalError, TileTypeError
import cuda.lang._datatype as datatype
from ..type_checking_helpers import require_vector_type
from ..op_defs import RawMLIROperation, VectorGetItem
from ..type import ScalarTy, Type, VectorTy
from ..ir import Var, add_operation


_registry = ImplRegistry()
impl = _registry.impl


def vector_impl_registry() -> ImplRegistry:
    return _registry


def vector_undef(res_type: Type):
    return add_operation(
        RawMLIROperation, res_type, op_name="llvm.mlir.undef", operands_=()
    )


def vector_setitem(vector: Var[VectorTy], key: int | Var[ScalarTy], value: Var[ScalarTy]):
    ty = require_vector_type(vector)
    if isinstance(key, int):
        key = strictly_typed_const(key, ScalarTy(datatype.int32))
    key = implicit_cast(key, datatype.int32, "vector setitem index to int32")
    value = implicit_cast(
        value, ty.element_dtype, "vector setitem cast RHS to value type"
    )
    return add_operation(
        RawMLIROperation,
        vector.get_type(),
        op_name="llvm.insertelement",
        operands_=(vector, value, key),
    )


def vector_elementwise_apply(
    callable: Callable[[Var[ScalarTy], ...], Var[ScalarTy]], *vectors
):
    vector_types = [require_vector_type(v) for v in vectors]
    if len(vectors) == 0:
        raise TileInternalError("Expected at least one vector")
    length = vector_types[0].length
    if not all(v.length == length for v in vector_types[1:]):
        raise TileInternalError("Expected all vectors to have same length")

    def apply_one(i: int):
        index = strictly_typed_const(i, ScalarTy(datatype.int32))
        operands = [vector_getitem(x, index) for x in vectors]
        element = callable(*operands)
        return element

    first_element = apply_one(0)
    element_type = first_element.get_type()
    if not isinstance(element_type, ScalarTy):
        raise TileInternalError(
            "Expected elementwise application of function to vector to "
            f"return a scalar but got {element_type}"
        )

    res = vector_undef(VectorTy(element_type.dtype, length))
    res = vector_setitem(res, 0, first_element)
    for i in range(1, length):
        element = apply_one(i)
        res = vector_setitem(res, i, element)

    return res


# the user can't call __setitem__ in kernel code because the semantics might be
# confusing. We could expose an insertelement-like operation that maps to the
# vector_setitem utility though.
@impl(operator.setitem, overload=(VectorTy, WILDCARD, WILDCARD))
def vector_setitem_impl(object: Var[VectorTy], key: Var, value: Var):
    raise TileTypeError("Vectors are immutable: item assignment is not supported")


@impl(operator.getitem, overload=(VectorTy, WILDCARD))
def vector_getitem(object: Var[VectorTy], key: Var[ScalarTy]) -> Var[ScalarTy]:
    result_dtype = object.get_type().element_dtype
    index = implicit_cast(key, datatype.int32, "vector getitem index")
    return add_operation(
        VectorGetItem,
        ScalarTy(result_dtype),
        x=object,
        index=index,
    )
