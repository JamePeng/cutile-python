# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import enum
import re
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Sequence, Any
from . import _codes as codes

from cuda.tile._cext import BitstreamWriter


DATALAYOUT_PTX = ("e-p:64:64:64-p3:32:32:32"
                  "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128"
                  "-f32:32:32-f64:64:64"
                  "-v16:16:16-v32:32:32-v64:64:64-v128:128:128"
                  "-n16:32:64-a:8:8")


class CallingConvention(enum.Enum):
    C = 0
    PTX_Kernel = 71
    PTX_Device = 72


class Linkage(enum.Enum):
    External = 0


# Matches `enum BinaryOpcodes` in LLVMBitCodes.h
class Binop(enum.Enum):
    ADD = 0
    SUB = 1
    MUL = 2
    UDIV = 3
    SDIV = 4
    UREM = 5
    SREM = 6
    SHL = 7
    LSHR = 8
    ASHR = 9
    AND = 10
    OR = 11
    XOR = 12


class _BitcodeWriter(BitstreamWriter):
    def __init__(self):
        self.fixed32(0xdec04342)
        self._block_stack = [_Block(abbrev_encoder=self.fixed2)]

    @contextmanager
    def block(self, block_id: int, new_abbrev_len: int):
        match new_abbrev_len:
            case 2: abbrev_encoder = self.fixed2
            case 3: abbrev_encoder = self.fixed3
            case 4: abbrev_encoder = self.fixed4
            case 8: abbrev_encoder = self.fixed8
            case _: raise ValueError("Unsupported abbreviation length")
        self.abbrev(codes.ENTER_SUBBLOCK)
        self.vbr8(block_id)
        self.vbr4(new_abbrev_len)
        len_pos = self.aligned_word(0)  # block length to be patched later
        self._block_stack.append(_Block(abbrev_encoder))
        try:
            yield
            self.abbrev(codes.END_BLOCK)
            end_pos = self.align_to_word()
            self.patch_word(len_pos, end_pos - len_pos - 1)
        finally:
            self._block_stack.pop()

    def unabbrev_record(self, code: int, *operands: int):
        self.abbrev(codes.UNABBREV_RECORD)
        self.vbr6(code)
        self.vbr6(len(operands))
        for x in operands:
            self.vbr6(x)

    def define_abbrev(self, *operands: int | str) -> int:
        ret = self._block_stack[-1].generate_abbrev_id()
        self.abbrev(codes.DEFINE_ABBREV)
        self.vbr5(len(operands))
        for op in operands:
            self.fixed1(is_literal := isinstance(op, int))
            if is_literal:
                self.vbr8(op)
            elif m := re.fullmatch("(fixed|vbr)([0-9]+)", op):
                self.fixed3({"fixed": 1, "vbr": 2}[m.group(1)])
                self.vbr5(int(m.group(2)))
            else:
                self.fixed3({"array": 3, "char6": 4, "blob": 5}[op])
        return ret

    def blob(self, data: bytearray):
        self.vbr6(len(data))
        self.raw_blob(data)

    def abbrev(self, abbrev_id: int):
        self._block_stack[-1].abbrev_encoder(abbrev_id)


@dataclass(frozen=True)
class Type:
    type_id: int


@dataclass(frozen=True)
class FunctionType(Type):
    return_ty: Type
    parameter_types: tuple[Type, ...]


@dataclass
class Value:
    type: Type
    id: int | None = None


@dataclass
class Metadata:
    id: int | None = None


class _TypeDict(dict):
    def __missing__(self, key):
        ty = Type(len(self))
        self[key] = ty
        return ty


class TypeTable:
    def __init__(self):
        self._map = _TypeDict()

    def integer(self, bitwidth: int) -> Type:
        return self._map[(codes.TYPE_CODE_INTEGER, bitwidth)]

    def pointer(self, address_space: int) -> Type:
        return self._map[(codes.TYPE_CODE_OPAQUE_POINTER, address_space)]

    @property
    def I32(self) -> Type:
        return self.integer(32)

    @property
    def P0(self) -> Type:
        return self.pointer(0)

    @property
    def VOID(self) -> Type:
        return self._map[(codes.TYPE_CODE_VOID,)]

    @property
    def F32(self) -> Type:
        return self._map[(codes.TYPE_CODE_FLOAT,)]

    def function(self, return_type: Type, param_types: Sequence[Type]) -> FunctionType:
        key = (codes.TYPE_CODE_FUNCTION,
               0,  # isvararg
               return_type.type_id,
               *(t.type_id for t in param_types))

        if key not in self._map:
            self._map[key] = FunctionType(len(self._map), return_type, tuple(param_types))
        return self._map[key]


class _StringTable(dict[bytes, tuple[int, int]]):
    def __init__(self):
        super().__init__()
        self._total = 0

    def __missing__(self, key: bytes) -> tuple[int, int]:
        assert isinstance(key, bytes)
        offset = self._total
        self._total += len(key)
        ret = (offset, len(key))
        self[key] = ret
        return ret


@dataclass(frozen=True)
class _ConstantRecord:
    value: Value
    record: tuple[int, ...]


class _ConstantTable:
    def __init__(self):
        self._table: dict[Type, list[_ConstantRecord]] = defaultdict(list)

    def get(self, const_value: Any, ty: Type) -> Value:
        ret = Value(ty)
        if isinstance(const_value, int):
            record = (codes.CST_CODE_INTEGER, _transform_signed_int(const_value))
        else:
            raise TypeError(f"Unsupported constant value type {type(const_value)}")

        self._table[ty].append(_ConstantRecord(ret, record))
        return ret


@dataclass(frozen=True)
class _NullableMetadata:
    metadata: Metadata | None


@dataclass(frozen=True)
class _MetadataRecord:
    metadata: Metadata
    record: tuple[int | Value | Metadata | _NullableMetadata, ...]


class MetadataTable:
    def __init__(self):
        self._strings: list[tuple[Metadata, bytes]] = []
        self._non_strings: list[_MetadataRecord] = []

    def _is_empty(self) -> bool:
        return len(self._strings) == 0 and len(self._non_strings) == 0

    def string(self, s: str):
        ret = Metadata()
        self._strings.append((ret, s.encode()))
        return ret

    def value_as_metadata(self, value: Value) -> Metadata:
        return self._record(codes.METADATA_VALUE, value.type.type_id, value)

    def node(self, *items: Metadata | None) -> Metadata:
        return self._record(codes.METADATA_NODE, *(_NullableMetadata(x) for x in items))

    def _record(self, *record: int | Value | Metadata) -> Metadata:
        ret = Metadata()
        self._non_strings.append(_MetadataRecord(ret, record))
        return ret


@dataclass
class Function:
    name: str
    value: Value
    calling_convention: CallingConvention
    linkage: Linkage
    parameters: tuple[Value, ...]
    num_terminators: int = 0
    terminated: bool = False
    local_constants: _ConstantTable = dataclasses.field(default_factory=_ConstantTable)
    local_metadata: MetadataTable = dataclasses.field(default_factory=MetadataTable)
    instruction_data: list[int | Value] = dataclasses.field(default_factory=list)
    instruction_formats: list[str] = dataclasses.field(default_factory=list)
    instruction_results: list[Value | None] = dataclasses.field(default_factory=list)

    @property
    def type(self) -> FunctionType:
        ty = self.value.type
        assert isinstance(ty, FunctionType)
        return ty

    @property
    def is_declaration(self) -> bool:
        return len(self.instruction_formats) == 0


class BitcodeBuilder:
    def __init__(self,
                 target_triple: str | None = None,
                 data_layout: str | None = None):
        self._target_triple = target_triple
        self._data_layout = data_layout
        self._global_constants = _ConstantTable()
        self._global_metadata = MetadataTable()
        self._functions: list[Function] = []
        self._type_table = TypeTable()
        self._cur_function: Function | None = None
        self._named_metadata: list[tuple[str, tuple[Metadata, ...]]] = []

    def build(self) -> bytes:
        return _serialize_module(self)

    @property
    def type_table(self) -> TypeTable:
        return self._type_table

    @property
    def metadata(self) -> MetadataTable:
        return (self._global_metadata if self._cur_function is None
                else self._cur_function.local_metadata)

    def append_named_metadata(self, name: str, *metadata: Metadata):
        assert self._cur_function is None
        self._named_metadata.append((name, metadata))

    def append_nvvm_version_metadata(self, major: int, minor: int):
        i32 = self._type_table.I32
        major = self.constant(major, i32)
        minor = self.constant(minor, i32)
        major = self.metadata.value_as_metadata(major)
        minor = self.metadata.value_as_metadata(minor)
        version_node = self.metadata.node(major, minor)
        self.append_named_metadata("nvvmir.version", version_node)

    @contextmanager
    def function(self,
                 name: str,
                 type: FunctionType,
                 *,
                 calling_convention: CallingConvention = CallingConvention.C,
                 linkage: Linkage = Linkage.External):
        assert isinstance(calling_convention, CallingConvention)
        assert isinstance(linkage, Linkage)
        func_value = Value(type)
        parameters = tuple(Value(ty) for ty in type.parameter_types)
        func = Function(name=name,
                        value=func_value,
                        calling_convention=calling_convention,
                        linkage=linkage,
                        parameters=parameters)
        assert self._cur_function is None, "Functions cannot be nested"
        self._cur_function = func
        try:
            yield func
        finally:
            self._cur_function = None
        self._functions.append(func)

    def constant(self, value: Any, type: Type) -> Value:
        table = (self._global_constants if self._cur_function is None
                 else self._cur_function.local_constants)
        return table.get(value, type)

    def binop(self, result_ty: Type, op: "Binop", lhs: Value, rhs: Value) -> Value:
        assert isinstance(op, Binop)
        return self._instruction(result_ty, codes.FUNC_CODE_INST_BINOP, "Vvi", lhs, rhs, op._value_)

    def load(self, result_ty: Type, ptr: Value, alignment: int, volatile: bool = False) -> Value:
        assert alignment & (alignment - 1) == 0
        return self._instruction(result_ty, codes.FUNC_CODE_INST_LOAD, "Viii", ptr,
                                 result_ty.type_id, alignment.bit_length(), int(bool(volatile)))

    def store(self, ptr: Value, value: Value, alignment: int, volatile: bool = False):
        assert alignment & (alignment - 1) == 0
        self._instruction(None, codes.FUNC_CODE_INST_STORE, "VVii",
                          ptr, value, alignment.bit_length(), int(bool(volatile)))

    def call(self, func_ty: FunctionType, callee: Value, args: tuple[Value, ...]) -> Value:
        return self._instruction(
            func_ty.return_ty, codes.FUNC_CODE_INST_CALL, "iiiV" + "v" * len(args),
            0,  # attribute list ID
            1 << codes.CALL_EXPLICIT_TYPE,  # flags
            func_ty.type_id,
            callee,
            *args
        )

    def get_element_ptr(self, element_ty: Type, ptr: Value, *indices: Value):
        return self._instruction(
            ptr.type,
            codes.FUNC_CODE_INST_GEP,
            "iiV" + "V" * len(indices),
            0,  # Flags
            element_ty.type_id,
            ptr,
            *indices
        )

    def ret(self, *values: Value):
        self._instruction(None, codes.FUNC_CODE_INST_RET, "V" * len(values), *values,
                          terminator=True)

    def _instruction(self, result_ty: Type | None, code: int,
                     format: str, *instruction: int | Value,
                     terminator: bool = False) -> Value | None:
        """
        Format syntax:
            i: an immediate int
            v: a Value
            V: an optionally typed Value (for handling forward references)
        """
        f = self._cur_function
        assert f is not None
        f.instruction_data.append(code)
        f.instruction_data.extend(instruction)
        f.instruction_formats.append(format)
        result = None if result_ty is None else Value(result_ty)
        f.instruction_results.append(result)
        f.terminated = terminator
        if terminator:
            f.num_terminators += 1
        return result


def _serialize_module(builder: BitcodeBuilder) -> bytes:
    writer = _BitcodeWriter()
    string_table = _StringTable()
    with writer.block(codes.MODULE_BLOCK, 2):
        # Module version
        writer.unabbrev_record(codes.MODULE_CODE_VERSION, 2)

        # Type table
        _write_type_table(builder.type_table, writer)

        # Module info
        if builder._target_triple is not None:
            writer.unabbrev_record(codes.MODULE_CODE_TRIPLE, *builder._target_triple.encode())
        if builder._data_layout is not None:
            writer.unabbrev_record(codes.MODULE_CODE_DATALAYOUT, *builder._data_layout.encode())

        # Global constants
        assign_value_id = _IdMapper(0)
        _write_constant_table(builder._global_constants, writer, assign_value_id)

        # Global metadata
        assign_metadata_id = _IdMapper(0)
        if not builder._global_metadata._is_empty() or len(builder._named_metadata) > 0:
            with writer.block(codes.METADATA_BLOCK, 3):
                _write_metadata_table(builder._global_metadata, writer, assign_metadata_id)
                _write_named_metadata(builder._named_metadata, writer)

        # Write function declarations
        for func in builder._functions:
            assign_value_id(func.value)
            _write_function_declaration_record(func, writer, string_table)

        # Write the functions
        for func in builder._functions:
            if func.is_declaration:
                continue
            assert func.terminated
            with (writer.block(codes.FUNCTION_BLOCK, 2),
                  assign_value_id.checkpoint(), assign_metadata_id.checkpoint()):
                for val in func.parameters:
                    assign_value_id(val)
                _write_constant_table(func.local_constants, writer, assign_value_id)
                if not func.local_metadata._is_empty():
                    with writer.block(codes.METADATA_BLOCK, 3):
                        _write_metadata_table(func.local_metadata, writer, assign_metadata_id)
                first_instruction_id = assign_value_id.next_id
                for val in func.instruction_results:
                    if val is not None:
                        assign_value_id(val)
                writer.unabbrev_record(codes.FUNC_CODE_DECLAREBLOCKS, func.num_terminators)
                _write_function_body(func, first_instruction_id, writer)

    _write_string_table(string_table, writer)
    return writer.to_bytes()


def _write_constant_table(constant_table: _ConstantTable, writer: _BitcodeWriter,
                          assign_value_id: "_IdMapper"):
    if len(constant_table._table) == 0:
        return
    with writer.block(codes.CONSTANTS_BLOCK, 2):
        for ty, constants in constant_table._table.items():
            writer.unabbrev_record(codes.CST_CODE_SETTYPE, ty.type_id)
            for const_record in constants:
                assign_value_id(const_record.value)
                writer.unabbrev_record(*const_record.record)


def _write_metadata_table(table: MetadataTable, writer: _BitcodeWriter,
                          assign_metadata_id: "_IdMapper"):
    if len(table._strings) > 0:
        blob_writer = BitstreamWriter()
        for metadata, s in table._strings:
            assign_metadata_id(metadata)
            blob_writer.vbr6(len(s))
        string_lengths = blob_writer.to_bytes()

        meta_strings_abbrev = writer.define_abbrev(
            codes.METADATA_STRINGS,
            "vbr6",  # number of strings
            "vbr6",  # offset of data inside the blob
            "blob"
        )

        blob = bytearray(string_lengths)
        for _metadata, s in table._strings:
            blob += s

        writer.abbrev(meta_strings_abbrev)
        writer.vbr6(len(table._strings))
        writer.vbr6(len(string_lengths))
        writer.blob(blob)

    for meta_rec in table._non_strings:
        assign_metadata_id(meta_rec.metadata)
        resolved_record = [_resolve_metadata_item(x) for x in meta_rec.record]
        writer.unabbrev_record(*resolved_record)


def _resolve_metadata_item(val: int | Value | Metadata | _NullableMetadata) -> int:
    if isinstance(val, int):
        return val
    elif isinstance(val, _NullableMetadata):
        if val.metadata is None:
            return 0
        else:
            return val.metadata.id + 1
    else:
        assert isinstance(val, Value | Metadata)
        assert val.id is not None
        return val.id


def _write_named_metadata(named_metadata: Sequence[tuple[str, tuple[Metadata, ...]]],
                          writer: _BitcodeWriter):
    if len(named_metadata) == 0:
        return

    name_abbrev = writer.define_abbrev(codes.METADATA_NAME, "array", "fixed8")
    for name, operands in named_metadata:
        # Name record
        name_utf8 = name.encode()
        writer.abbrev(name_abbrev)
        writer.vbr6(len(name_utf8))
        for c in name_utf8:
            writer.fixed8(c)

        # Node record
        writer.unabbrev_record(codes.METADATA_NAMED_NODE, *(m.id for m in operands))


def _transform_signed_int(x: int) -> int:
    return x << 1 if x >= 0 else (-x << 1) | 1


def _write_string_table(string_table: _StringTable, writer: _BitcodeWriter):
    # Concatenate all strings into a single bytearray
    data = bytearray(string_table._total)
    running_offset = 0
    for string, (offset, length) in string_table.items():
        assert running_offset == offset
        assert len(string) == length
        running_offset += length
        data[offset:running_offset] = string
    assert running_offset == len(data)

    # Write the STRTAB block
    with writer.block(codes.STRTAB_BLOCK, 3):
        abbrev_id = writer.define_abbrev(codes.STRTAB_BLOB, "blob")
        writer.abbrev(abbrev_id)
        writer.blob(data)


def _write_type_table(type_table: TypeTable, writer: _BitcodeWriter):
    with writer.block(codes.TYPE_BLOCK, 2):
        writer.unabbrev_record(codes.TYPE_CODE_NUMENTRY, len(type_table._map))
        for i, (rec, ty) in enumerate(type_table._map.items()):
            assert i == ty.type_id
            writer.unabbrev_record(*rec)


def _write_function_body(func: Function, first_instruction_id: int, writer: _BitcodeWriter):
    data_iter = iter(func.instruction_data)
    instruction_id = first_instruction_id
    for format, result in zip(func.instruction_formats, func.instruction_results,
                              strict=True):
        code = next(data_iter)
        assert isinstance(code, int)
        operands = []
        for f in format:
            operand = next(data_iter)
            if f == "i":
                assert isinstance(operand, int)
                operands.append(operand)
            else:
                assert f in "vV"
                assert isinstance(operand, Value)
                assert operand.id is not None
                relative_id = instruction_id - operand.id
                operands.append(relative_id & 0xffff_ffff)
                if f == "V" and relative_id <= 0:
                    operands.append(operand.type.type_id)
        writer.unabbrev_record(code, *operands)
        if result is not None:
            instruction_id += 1

    assert len(list(data_iter)) == 0


def _write_function_declaration_record(func: Function, writer: _BitcodeWriter,
                                       string_table: _StringTable):
    writer.unabbrev_record(
        codes.MODULE_CODE_FUNCTION,
        *string_table[func.name.encode()],  # STRTAB offset & size
        func.value.type.type_id,
        func.calling_convention._value_,
        func.is_declaration,
        func.linkage._value_,
        0,  # attributes
        0,  # alignment
        0,  # section
        0,  # default visibility
        0,  # GC
        0,  # unnamed_addr
        0,  # prologue_data
        0,  # DLL storage class
        0,  # comdat
        0,  # prefix data
        0,  # personality function
        0,  # DSO local
    )


@dataclass
class _IdMapper:
    next_id: int

    @contextmanager
    def checkpoint(self):
        old = self.next_id
        try:
            yield
        finally:
            self.next_id = old

    def __call__(self, value: Value | Metadata):
        if value.id is None:
            value.id = self.next_id
        else:
            assert value.id == self.next_id
        self.next_id += 1


@dataclass
class _Block:
    abbrev_encoder: Callable[[int], None]
    next_abbrev_id: int = 4

    def generate_abbrev_id(self) -> int:
        ret = self.next_abbrev_id
        self.next_abbrev_id += 1
        return ret
