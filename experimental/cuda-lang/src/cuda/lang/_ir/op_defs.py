# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import cuda.lang._mlir as mlir
from cuda.lang._ir.ir import Operation, Var, attribute, operand
from cuda.tile._ir.ir import MemoryEffect


@dataclass(eq=False)
class RawNVVMIntrinsic(
    Operation, opcode="nvvm.call_intrinsic", memory_effect=MemoryEffect.STORE
):
    intrinsic: str = attribute()
    operands_: tuple[Var, ...] = operand()


@dataclass(eq=False)
class RawMLIROperation(Operation, opcode="mlir.operation",
                       memory_effect=MemoryEffect.STORE):
    op_name: str = attribute()
    operands_: tuple[Var, ...] = operand()
    mlir_attributes: tuple[tuple[str, mlir.Attribute], ...] = attribute(default=())
