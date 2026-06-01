# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
import dataclasses
from dataclasses import dataclass
from types import MethodType, FunctionType, BuiltinFunctionType
from typing import Any, Optional

from typing_extensions import override

import cuda.tile._bytecode as bc
from cuda.tile import TileTypeError
from cuda.tile._datatype import numeric_dtype_category
from cuda.tile._ir import hir_stubs
from cuda.tile._ir.ir import Operation, attribute, Var, Builder, make_aggregate
from cuda.tile._ir.op_impl import ImplRegistry, require_dataclass_type
from cuda.tile._ir.type import Type, DTypeSpec, TensorLikeTy, TupleTy, TupleValue, Symbol, \
    DataclassInfo, DataclassTy, DataclassValue, BoundMethodValue, BoundMethodTy
from cuda.tile._ir.typing_support import type_of_constant_python_value, \
    loose_type_of_constant_python_value, get_dataclass_info, as_third_party_dtype_spec
from cuda.tile._ir2bytecode import BytecodeContext


_registry = ImplRegistry()
impl = _registry.impl


def core_impl_registry() -> ImplRegistry:
    return _registry


@dataclass(eq=False)
class TypedConst(Operation, opcode="typed_const"):
    value: Any = attribute()

    @override
    def generate_bytecode(self, ctx: BytecodeContext) -> bc.Value:
        return ctx.constant(self.value, ctx.typeof(self.result_var))


def loosely_typed_const(value: Any,
                        ty: Optional[Type] = None,
                        loose_ty: Optional[Type] = None,
                        name: str | None = None) -> Var:
    builder = Builder.get_current()
    if ty is None:
        ty = type_of_constant_python_value(value, builder.ir_ctx.typing_hooks)
    assert not ty.is_aggregate(), "Use sym2var(value, constant_only=True) instead"

    # Normalize third party dtype spec objects (e.g. torch.float32 -> ct.float32)
    if isinstance(ty, DTypeSpec):
        value = ty.dtype

    ret = _strictly_typed_const_inner(builder, value, ty, name=name)
    if loose_ty is None:
        loose_ty = loose_type_of_constant_python_value(value, builder.ir_ctx.typing_hooks)
    ret.set_loose_type(loose_ty)
    return ret


def strictly_typed_const(value: Any, ty: Type, name: str | None = None) -> Var:
    return _strictly_typed_const_inner(Builder.get_current(), value, ty, name)


def _strictly_typed_const_inner(builder: Builder,
                                value: Any, ty: Type, name: str | None = None) -> Var:
    result = None if name is None else builder.ir_ctx.make_var(name, builder.loc)
    ret = builder.add_operation(TypedConst, ty, dict(value=value), result=result)
    if not isinstance(ty, TensorLikeTy) or ty.tensor_shape() == ():
        # We currently don't have a way to represent an N-dimensional tile constant
        ret.set_constant(value)
    return ret


@impl(hir_stubs.build_tuple)
def build_tuple(items: tuple[Var, ...]) -> Var:
    ty = TupleTy(tuple(x.get_type() for x in items))
    loose_ty = TupleTy(tuple(x.get_loose_type() for x in items))
    res = make_aggregate(TupleValue(items), ty, loose_ty)
    if all(x.is_constant() for x in items):
        res.set_constant(tuple(x.get_constant() for x in items))
    return res


def build_dataclass_instance(items: tuple[Var, ...], info: DataclassInfo) -> Var:
    cls = info.cls
    ty = DataclassTy(cls, tuple(x.get_type() for x in items))
    loose_ty = DataclassTy(cls, tuple(x.get_loose_type() for x in items))
    res = make_aggregate(DataclassValue(items, info), ty, loose_ty)
    if all(x.is_constant() for x in items):
        const_val = cls(**{name: x.get_constant()
                           for name, x in zip(info.field_names, items, strict=True)})
        res.set_constant(const_val)
    return res


@impl(dataclasses.replace)
def dataclasses_replace_impl(obj: Var, changes: dict[str, Var]):
    dataclass_ty = require_dataclass_type(obj)
    dataclass_val = obj.get_aggregate()
    assert isinstance(dataclass_val, DataclassValue)
    name2idx = dataclass_val.info.field_name_to_idx
    new_items = list(dataclass_val.items)
    for name, val in changes.items():
        try:
            idx = name2idx[name]
        except KeyError:
            raise TileTypeError(f"Dataclass '{dataclass_ty.cls.__name__}'"
                                f" has no such field '{name}'")
        new_items[idx] = val
    return build_dataclass_instance(tuple(new_items), dataclass_val.info)


def bind_method(object: Var, func) -> Var:
    agg_value = BoundMethodValue(object)
    res_ty = BoundMethodTy(object.get_type(), func)
    return make_aggregate(agg_value, res_ty)


def sym2var(x: Any, constant_only: bool = False) -> Var:
    # TODO: verify we don't have a stale closure

    if isinstance(x, Symbol):
        if constant_only:
            raise TileTypeError("Cannot create a constant from a symbolic value")
        return x._var

    if isinstance(x, tuple):
        return build_tuple(tuple(sym2var(item, constant_only=constant_only) for item in x))

    cls = type(x)
    if dataclasses.is_dataclass(cls):
        info = get_dataclass_info(cls)
        field_vars = tuple(sym2var(getattr(x, f.name), constant_only=constant_only)
                           for f in dataclasses.fields(cls))
        return build_dataclass_instance(field_vars, info)

    if isinstance(x, MethodType):
        self_var = sym2var(x.__self__, constant_only=constant_only)
        if not isinstance(x.__func__, FunctionType | BuiltinFunctionType):
            raise TileTypeError(f"Object of type {type(x).__name__}"
                                f" cannot be used as a function for binding a method")
        return bind_method(self_var, x.__func__)

    # Transform a third party typed scalar (e.g., np.int16(5)) into a strictly typed constant
    dtype_spec = as_third_party_dtype_spec(type(x))
    if dtype_spec is not None:
        pyval = numeric_dtype_category(dtype_spec.dtype).pytype(x)
        ty = Builder.get_current().ir_ctx.typing_hooks.get_tensor_like_type(dtype_spec.dtype, ())
        return strictly_typed_const(pyval, ty)

    return loosely_typed_const(x)
