# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Frontend typing and IR construction for compiled-host kernel launches."""

from cuda.tile._ir.aggregate_support import flatten_aggregates
from cuda.tile._ir.core_ops import loosely_typed_const, strictly_typed_const
from cuda.tile._ir.ir import Var, add_operation_variadic
from cuda.tile._ir.op_impl import ImplRegistry, require_constant_bool
from cuda.tile._ir.type import TupleTy, TupleValue

import cuda.lang._datatype as datatype
from cuda.lang._execution import kernel as Kernel, launch
from cuda.lang._exception import TypeCheckingError
from cuda.lang._ir.op_defs import KernelLaunch
from cuda.lang._ir.type import ScalarTy, StreamTy


_registry = ImplRegistry()
impl = _registry.impl


def host_impl_registry() -> ImplRegistry:
    return _registry


def require_launch_dimensions(
    value: Var,
    parameter_name: str,
    *,
    optional: bool = False,
) -> tuple[Var, ...] | None:
    if optional and value.is_constant() and value.get_constant() is None:
        return None
    if not isinstance(value.get_type(), TupleTy):
        raise TypeCheckingError(f"{parameter_name} must be a tuple")
    aggregate = value.get_aggregate()
    assert isinstance(aggregate, TupleValue)
    dimensions = aggregate.items
    if not 1 <= len(dimensions) <= 3:
        raise TypeCheckingError(f"{parameter_name} must contain one to three values")
    return dimensions


def require_stream(value: Var) -> Var:
    if value.is_constant() and value.get_constant() is None:
        return strictly_typed_const(0, ScalarTy(datatype.int64))
    ty = value.get_type()
    if isinstance(ty, StreamTy):
        return value
    if not isinstance(ty, ScalarTy) or ty.dtype is not datatype.int64:
        raise TypeCheckingError(
            "compiled host cl.launch() stream must be a CUDA stream compatible type, "
            "None, or an int64 raw handle."
        )
    return value


@impl(launch)
def launch_impl(
    stream: Var,
    block_count: Var,
    thread_count: Var,
    kernel: Var,
    kernel_args: Var,
    cooperative: Var,
    block_in_cluster_count: Var,
    preferred_block_in_cluster_count: Var,
    programmatic_dependent_launch: Var,
) -> Var:
    stream = require_stream(stream)
    if not kernel.is_constant() or not isinstance(kernel.get_constant(), Kernel):
        raise TypeCheckingError("cl.launch() requires a statically known cuda.lang kernel")
    if not isinstance(kernel_args.get_type(), TupleTy):
        raise TypeCheckingError("cl.launch() kernel_args must be a tuple")
    args_value = kernel_args.get_aggregate()
    assert isinstance(args_value, TupleValue)
    target = kernel.get_constant()
    if len(args_value.items) != len(target._annotated_function.pysig.parameters):
        raise TypeCheckingError("cl.launch() argument count does not match the target kernel")

    logical_args = args_value.items
    logical_types = tuple(argument.get_type() for argument in logical_args)

    add_operation_variadic(
        KernelLaunch,
        (),
        stream=stream,
        block_count=require_launch_dimensions(block_count, "block_count"),
        thread_count=require_launch_dimensions(thread_count, "thread_count"),
        kernel_argument_leaves=flatten_aggregates(logical_args, logical_types),
        kernel_argument_types=logical_types,
        launched_kernel=target,
        cooperative=require_constant_bool(cooperative),
        block_in_cluster_count=require_launch_dimensions(
            block_in_cluster_count, "block_in_cluster_count", optional=True
        ),
        preferred_block_in_cluster_count=require_launch_dimensions(
            preferred_block_in_cluster_count,
            "preferred_block_in_cluster_count",
            optional=True,
        ),
        programmatic_dependent_launch=require_constant_bool(
            programmatic_dependent_launch
        ),
    )
    return loosely_typed_const(None)
