# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from typing import Sequence

from cuda.lang._ir import ir
from cuda.lang._ir import type as ir_type
from cuda.lang._target import TargetInfo

from .. import _llvm_bitcode as llvm
from .._llvm_bitcode import BitcodeBuilder, DATALAYOUT_PTX, Cast, CmpPredicate, Binop
from cuda.lang import _datatype as datatype
from cuda.tile._numeric_semantics import RoundingMode
from cuda.tile._bytecode import float_to_bits
from cuda.tile._datatype import dtype_simple_bytecode_type, is_integral, is_signed, is_boolean


DIRECTLY_SUPPORTED_FLOATS = {
    datatype.float16: llvm.FloatKind.f16,
    datatype.bfloat16: llvm.FloatKind.bf16,
    datatype.float32: llvm.FloatKind.f32,
    datatype.float64: llvm.FloatKind.f64,
}


def dtype_to_llvm(dtype: datatype.DType, tt: llvm.TypeTable,
                  storage: bool) -> llvm.Type:
    if datatype.is_pointer_dtype(dtype):
        info = datatype.PointerInfo(dtype)
        return tt.pointer(info.memory_space._value_)
    elif dtype == datatype.bool_:
        return tt.integer(8 if storage else 1)
    elif is_integral(dtype):
        return tt.integer(dtype.bitwidth)
    elif (kind := DIRECTLY_SUPPORTED_FLOATS.get(dtype)) is not None:
        return tt.float(kind)
    elif datatype.is_float(dtype):
        return tt.integer((dtype.bitwidth + 7) // 8 if storage else dtype.bitwidth)
    elif dtype == datatype.mbarrier:
        return tt.integer(64)
    elif dtype == datatype.cluster_launch_control_token:
        return tt.integer(128)
    else:
        raise NotImplementedError(f"Unsupported dtype {dtype}")


def type_to_llvm(ty: ir_type.Type, tt: llvm.TypeTable, storage: bool) -> llvm.Type:
    if isinstance(ty, ir_type.ScalarTy):
        return dtype_to_llvm(ty.dtype, tt, storage)
    elif isinstance(ty, ir_type.PointerTy):
        return dtype_to_llvm(ty.pointer_dtype, tt, storage)
    elif isinstance(ty, ir_type.VectorTy):
        el_ty = dtype_to_llvm(ty.element_dtype, tt, storage)
        return tt.vector(el_ty, ty.length)
    elif isinstance(ty, ir_type.TensorMapTy):
        return tt.pointer(0)
    else:
        raise NotImplementedError(f"Unsupported type {ty}")


class LLVMLoweringContext:
    def __init__(self,
                 builder: BitcodeBuilder,
                 ir_ctx: ir.IRContext,
                 target_info: TargetInfo | None,  # None for host
                 ):
        self.ir_ctx = ir_ctx
        self.builder = builder
        self.target_info = target_info
        self._value_map: dict[str, llvm.Value] = dict()
        self._used_instrinsics: dict[str, llvm.Value] = dict()

    def value(self, var: ir.Var) -> llvm.Value:
        return self._value_map[var.name]

    def set_value(self, var: ir.Var, value: llvm.Value):
        name = var.name
        if name in self._value_map:
            raise ValueError(f"Variable {name} is already in the value map")
        self._value_map[name] = value

    def typeof(self, var: ir.Var) -> llvm.Type:
        return type_to_llvm(var.get_type(), self.builder.type_table, storage=False)

    def dtype(self, dtype: datatype.DType, storage: bool) -> llvm.Type:
        return dtype_to_llvm(dtype, self.builder.type_table, storage=storage)

    def type(self, type: ir_type.Type, storage: bool) -> llvm.Type:
        return type_to_llvm(type, self.builder.type_table, storage=storage)

    def constant(self, value: bool | int | float, ty: ir_type.TensorLikeTy) -> llvm.Value:
        dtype = ty.tensor_dtype()
        llvm_ty = type_to_llvm(ty, self.builder.type_table, storage=False)
        if is_boolean(dtype) or is_integral(dtype):
            value = int(value)
            if value < 0:
                value += 1 << dtype.bitwidth
            return self.builder.constants.integer_constant(value, llvm_ty)
        elif datatype.is_float(dtype):
            value = float(value)
            # TODO: make float_to_bits() etc. independent of the tile bytecode enums
            bits = float_to_bits(value, dtype_simple_bytecode_type(dtype))
            if dtype in DIRECTLY_SUPPORTED_FLOATS:
                return self.builder.constants.float_constant(bits, llvm_ty)
            else:
                return self.builder.constants.integer_constant(bits, llvm_ty)
        else:
            raise NotImplementedError(f"Cannot create a constant of type {ty}")

    def cast(self,
             value: llvm.Value,
             from_type: ir_type.TensorLikeTy,
             to_type: ir_type.TensorLikeTy,
             rounding_mode: RoundingMode | None = None) -> llvm.Value:
        assert from_type.tensor_shape() == to_type.tensor_shape()
        from_dtype = from_type.tensor_dtype()
        to_dtype = to_type.tensor_dtype()
        assert datatype.is_numeric(from_dtype)
        assert datatype.is_numeric(to_dtype)
        if from_dtype == to_dtype:
            return value

        res_ty = type_to_llvm(to_type, self.builder.type_table, storage=False)

        # Handle integer-to-integer and boolean-to-integer first
        if ((is_integral(from_dtype) or is_boolean(from_dtype))
                and is_integral(to_dtype)):
            if from_dtype.bitwidth == to_dtype.bitwidth:
                return value
            elif from_dtype.bitwidth > to_dtype.bitwidth:
                return self.builder.cast(res_ty, Cast.TRUNC, value)
            elif datatype.is_signed(from_dtype):
                return self.builder.cast(res_ty, Cast.SEXT, value)
            else:
                return self.builder.cast(res_ty, Cast.ZEXT, value)

        # Integer/float to boolean
        if is_boolean(to_dtype):
            zero = self.constant(0, from_type)
            predicate = (CmpPredicate.FCMP_UNE if from_dtype in DIRECTLY_SUPPORTED_FLOATS
                         else CmpPredicate.ICMP_NE)
            return self.builder.cmp(predicate, value, zero)

        if rounding_mode is not None:
            raise NotImplementedError("Non-default rounding mode is not supported")

        # Direct float-to-float
        if from_dtype in DIRECTLY_SUPPORTED_FLOATS and to_dtype in DIRECTLY_SUPPORTED_FLOATS:
            if from_dtype.bitwidth > to_dtype.bitwidth:
                return self.builder.cast(res_ty, Cast.FPTRUNC, value)
            elif from_dtype.bitwidth < to_dtype.bitwidth:
                return self.builder.cast(res_ty, Cast.FPEXT, value)
            else:
                assert from_dtype in (datatype.float16, datatype.bfloat16)
                assert to_dtype in (datatype.float16, datatype.bfloat16)
                f32_ty = self.ir_ctx.typing_hooks.get_tensor_like_type(
                    datatype.float32, from_type.tensor_shape())
                f32_value = self.builder.cast(
                    type_to_llvm(f32_ty, self.builder.type_table, storage=False), Cast.FPEXT, value)
                return self.cast(f32_value, f32_ty, to_type, rounding_mode)

        # Direct float to integer
        if from_dtype in DIRECTLY_SUPPORTED_FLOATS and is_integral(to_dtype):
            cast = Cast.FPTOSI if datatype.is_signed(to_dtype) else Cast.FPTOUI
            return self.builder.cast(res_ty, cast, value)

        # Int/bool to direct float
        if ((is_integral(from_dtype) or is_boolean(from_dtype))
                and to_dtype in DIRECTLY_SUPPORTED_FLOATS):
            cast = Cast.SITOFP if datatype.is_signed(from_dtype) else Cast.UITOFP
            return self.builder.cast(res_ty, cast, value)

        raise NotImplementedError(f"Unsupported type conversion"
                                  f" from {from_dtype} to {to_dtype}")

    def floor(self, type: ir_type.TensorLikeTy, x: llvm.Value) -> llvm.Value:
        return self.call_intrinsic("llvm.floor", (type,), (type,), (x,))[0]

    def binary_arithmetic(self,
                          fn: str,
                          type: ir_type.TensorLikeTy,
                          lhs: llvm.Value,
                          rhs: llvm.Value,
                          *,
                          rounding_mode: RoundingMode | None = None,
                          flush_to_zero: bool = False,
                          propagate_nan: bool = True) -> llvm.Value:
        simple = rounding_mode is None and not flush_to_zero
        if not simple:
            raise NotImplementedError()
        dtype = type.tensor_dtype()
        match fn:
            case "add": return self.builder.binop(Binop.ADD, lhs, rhs)
            case "sub": return self.builder.binop(Binop.SUB, lhs, rhs)
            case "mul": return self.builder.binop(Binop.MUL, lhs, rhs)
            case "floordiv" if datatype.is_float(dtype):
                tmp = self.builder.binop(Binop.SDIV, lhs, rhs)
                return self.floor(type, tmp)
            case ("floordiv" | "cdiv") if is_integral(dtype) and datatype.is_signed(dtype):
                # q = SDIV(lhs, rhs)
                # p = q * rhs
                # if p != lhs & (lhs < 0) ==/!= (rhs < 0)  # depending on cdiv/floordiv
                #    result = q +/- 1  # depending on cdiv/floordiv
                # else:
                #    result = q
                q = self.builder.binop(Binop.SDIV, lhs, rhs)
                p = self.builder.binop(Binop.MUL, q, rhs)
                ne = self.builder.cmp(CmpPredicate.ICMP_NE, lhs, p)
                zero = self.constant(0, type)
                lhs_neg = self.builder.cmp(CmpPredicate.ICMP_SLT, lhs, zero)
                rhs_neg = self.builder.cmp(CmpPredicate.ICMP_SLT, rhs, zero)
                cmp, correction = ((CmpPredicate.ICMP_NE, -1), (CmpPredicate.ICMP_EQ, 1))[
                    ("floordiv", "cdiv").index(fn)]
                need_correction = self.builder.cmp(cmp, lhs_neg, rhs_neg)
                cond = self.builder.binop(Binop.AND, ne, need_correction)
                neg_one = self.constant(correction, type)
                q_minus_one = self.builder.binop(Binop.ADD, q, neg_one)
                return self.builder.select(cond, q_minus_one, q)
            case "floordiv" if is_integral(dtype) and not datatype.is_signed(dtype):
                return self.builder.binop(Binop.UDIV, lhs, rhs)
            case "cdiv" if is_integral(dtype) and not datatype.is_signed(dtype):
                # 0 if lhs == 0 else UDIV(lhs - 1, m) + 1
                zero = self.constant(0, type)
                is_zero = self.builder.cmp(CmpPredicate.ICMP_EQ, lhs, zero)
                one = self.constant(1, type)
                lhs_minus_one = self.builder.binop(Binop.SUB, lhs, one)
                q = self.builder.binop(Binop.UDIV, lhs_minus_one, rhs)
                res = self.builder.binop(Binop.ADD, q, one)
                return self.builder.select(is_zero, zero, res)
            case "truediv" if datatype.is_float(dtype):
                return self.builder.binop(Binop.SDIV, lhs, rhs)
            case "c_mod" if datatype.is_signed(dtype):
                return self.builder.binop(Binop.SREM, lhs, rhs)
            case "c_mod" if not datatype.is_signed(dtype):
                return self.builder.binop(Binop.UREM, lhs, rhs)
            case "xor": return self.builder.binop(Binop.XOR, lhs, rhs)
            case "or_": return self.builder.binop(Binop.OR, lhs, rhs)
            case "and_": return self.builder.binop(Binop.AND, lhs, rhs)
            case _:
                raise NotImplementedError(
                        f"Missing binary arithmetic implementation for"
                        f" {fn}[{type}, {rounding_mode}, ftz={flush_to_zero}]")

    def comparison(self,
                   fn: str,
                   type: ir_type.TensorLikeTy,
                   lhs: llvm.Value,
                   rhs: llvm.Value):
        dtype = type.tensor_dtype()
        match fn:
            case "eq" if dtype in DIRECTLY_SUPPORTED_FLOATS: pred = CmpPredicate.FCMP_OEQ
            case "eq" if is_integral(dtype) or is_boolean(dtype): pred = CmpPredicate.ICMP_EQ

            case "ne" if dtype in DIRECTLY_SUPPORTED_FLOATS: pred = CmpPredicate.FCMP_ONE
            case "ne" if is_integral(dtype) or is_boolean(dtype): pred = CmpPredicate.ICMP_NE

            case "ge" if dtype in DIRECTLY_SUPPORTED_FLOATS: pred = CmpPredicate.FCMP_OGE
            case "ge" if is_integral(dtype) and is_signed(dtype): pred = CmpPredicate.ICMP_SGE
            case "ge" if _is_uint_or_bool(dtype): pred = CmpPredicate.ICMP_UGE

            case "gt" if dtype in DIRECTLY_SUPPORTED_FLOATS: pred = CmpPredicate.FCMP_OGT
            case "gt" if is_integral(dtype) and is_signed(dtype): pred = CmpPredicate.ICMP_SGT
            case "gt" if _is_uint_or_bool(dtype): pred = CmpPredicate.ICMP_UGT

            case "ge" if dtype in DIRECTLY_SUPPORTED_FLOATS: pred = CmpPredicate.FCMP_OGE
            case "ge" if is_integral(dtype) and is_signed(dtype): pred = CmpPredicate.ICMP_SGE
            case "ge" if _is_uint_or_bool(dtype): pred = CmpPredicate.ICMP_UGE

            case "lt" if dtype in DIRECTLY_SUPPORTED_FLOATS: pred = CmpPredicate.FCMP_OLT
            case "lt" if is_integral(dtype) and is_signed(dtype): pred = CmpPredicate.ICMP_SLT
            case "lt" if _is_uint_or_bool(dtype): pred = CmpPredicate.ICMP_ULT

            case "le" if dtype in DIRECTLY_SUPPORTED_FLOATS: pred = CmpPredicate.FCMP_OLE
            case "le" if is_integral(dtype) and is_signed(dtype): pred = CmpPredicate.ICMP_SLE
            case "le" if _is_uint_or_bool(dtype): pred = CmpPredicate.ICMP_ULE

            case _:
                raise NotImplementedError(f"Missing comparison implementation for {fn}[{type}]")

        return self.builder.cmp(pred, lhs, rhs)

    def bitwise_shift(self,
                      fn: str,
                      type: ir_type.TensorLikeTy,
                      lhs: llvm.Value,
                      rhs: llvm.Value):
        dtype = type.tensor_dtype()
        match fn:
            case "lshift": op = Binop.SHL
            case "rshift" if is_signed(dtype): op = Binop.ASHR
            case "rshift" if not is_signed(dtype): op = Binop.LSHR
            case _:
                raise NotImplementedError(f"Missing bit shift implementation for {fn}[{type}]")
        return self.builder.binop(op, lhs, rhs)

    def binary_bitwise(self,
                       fn: str,
                       lhs: llvm.Value,
                       rhs: llvm.Value):
        match fn:
            case "and_": op = Binop.AND
            case "or_": op = Binop.OR
            case "xor": op = Binop.XOR
            case _:
                raise NotImplementedError(f"Missing bitwise binary implementation for {fn}")
        return self.builder.binop(op, lhs, rhs)

    def call_intrinsic(self,
                       base_name: str,
                       return_types: Sequence[ir_type.Type],
                       param_types: Sequence[ir_type.Type | None],
                       args: Sequence[llvm.Value | llvm.Metadata]) -> tuple[llvm.Value, ...]:
        # Create an llvm.FunctionType for the signature
        tt = self.builder.type_table
        if len(return_types) == 0:
            ret_ty_llvm = tt.VOID
        elif len(return_types) == 1:
            ret_ty_llvm = self.type(return_types[0], storage=False)
        else:
            ret_ty_llvm = tt.struct_anonymous(
                    [self.type(t, storage=False) for t in return_types])
        param_types_llvm = [self.builder.type_table.METADATA if t is None
                            else self.type(t, storage=False)
                            for t in param_types]
        func_ty = tt.function(ret_ty_llvm, param_types_llvm)

        # Generate a declaration if necessary
        from .._stub._nvvm_support import mangle_intrinsic_name
        mangled_name = mangle_intrinsic_name(base_name, param_types)
        if mangled_name in self._used_instrinsics:
            callee = self._used_instrinsics[mangled_name]
            assert callee.type.type_id == func_ty.type_id
        else:
            with self.builder.global_scope(), self.builder.function(mangled_name, func_ty) as f:
                callee = f.value
            self._used_instrinsics[mangled_name] = callee

        # Call the intrinsic
        result = self.builder.call(func_ty, callee, args)

        # Unpack the result
        if len(return_types) == 0:
            return ()
        elif len(return_types) == 1:
            return (result,)
        else:
            return tuple(self.builder.extract_value(self.type(t, storage=False), result, i)
                         for i, t in enumerate(return_types))


def _is_uint_or_bool(dtype: datatype.DType) -> bool:
    return (is_integral(dtype) and not is_signed(dtype)) or is_boolean(dtype)


def generate_nvvm_bitcode_for_kernel(body: ir.Region,
                                     symbol: str,
                                     target_info: TargetInfo) -> bytes:
    assert isinstance(target_info, TargetInfo)
    builder = BitcodeBuilder(target_triple="nvptx64-nvidia-cuda", data_layout=DATALAYOUT_PTX)
    ctx = LLVMLoweringContext(builder=builder,
                              ir_ctx=body.ctx,
                              target_info=target_info)
    builder.append_nvvm_version_metadata(2, 0)
    _lower_function(body=body,
                    name=symbol,
                    calling_convention=llvm.CallingConvention.PTX_Kernel,
                    ctx=ctx)
    return builder.build()


def _lower_function(body: ir.Region,
                    name: str,
                    calling_convention: llvm.CallingConvention,
                    ctx: LLVMLoweringContext,
                    ) -> llvm.Function:
    tt = ctx.builder.type_table
    func_ty = tt.function(tt.VOID, [ctx.typeof(p) for p in body.blocks[0].params])
    with ctx.builder.function(name, func_ty, calling_convention=calling_convention) as func:
        for param, llvm_value in zip(body.blocks[0].params, func.parameters, strict=True):
            ctx.set_value(param, llvm_value)

        for block in body.blocks:
            for op in block:
                result_values = op.generate_llvm(ctx)

                if isinstance(result_values, llvm.Value):
                    result_values = (result_values,)
                elif result_values is None:
                    result_values = ()

                for result_var, val in zip(op.result_vars, result_values, strict=True):
                    assert isinstance(val, llvm.Value)
                    ctx.set_value(result_var, val)
    return func
