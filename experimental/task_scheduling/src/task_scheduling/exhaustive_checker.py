# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded exhaustive interleaving checker for captured schedules."""

import itertools
from collections import deque
from dataclasses import dataclass, field, replace

from .enums import (
    FIRST_ITER,
    LAST_ITER,
    BlockGuard,
    Every,
    IterationPredicate,
    OpaqueCondition,
    PipelineType,
    SKIPPABLE,
    ScheduleStage,
    guard_fires,
    iter_leaves,
)
from .schedule_builder import (
    BreakLoop,
    ConditionalBlock,
    DomainLoop,
    Step,
    WhileTrueLoop,
    WorkTileLoop,
    _format_guard,
    _iter_nodes,
    validate_break_placement,
    while_loop_break_clauses,
)
from .resources import WorkQueue
from .task import Task


@dataclass(frozen=True)
class FlatScheduleOp:
    resource: object
    schedule_stage: ScheduleStage
    unique_id: int
    phase: str
    domain_start: object | None = None
    domain_end: object | None = None
    domain_step: object | None = None
    domain_entry: int | None = None
    sync_phase: str | None = None


def _bound(value: object, task: Task, fallback: int) -> int:
    if type(value) is int:
        return value
    if callable(value):
        try:
            folded = value(task, (0, 0, 0))
        except (AttributeError, TypeError):
            return fallback
        if type(folded) is int:
            return folded
    return fallback


def _domain_trip_count(
    node: DomainLoop,
    task: Task,
    dynamic_domain_fallback: int,
    representative_domain: bool,
) -> int:
    start = _bound(node.start, task, 0)
    end = (
        dynamic_domain_fallback
        if representative_domain
        else _bound(node.end, task, dynamic_domain_fallback)
    )
    stride = _bound(node.step, task, 1)
    if stride <= 0:
        raise ValueError("exhaustive checker requires positive domain step")
    return len(range(start, end, stride))


def _guard_phase(guard: object, phase: str) -> str:
    """Return the trace phase tag for operations inside a guarded block."""
    leaves = tuple(iter_leaves(guard))
    if any(isinstance(leaf, OpaqueCondition) for leaf in leaves):
        return "I"
    if any(leaf is FIRST_ITER for leaf in leaves):
        return "F"
    if any(leaf is LAST_ITER for leaf in leaves):
        return "L"
    if any(isinstance(leaf, Every) for leaf in leaves):
        return "P"
    return phase


OpaqueAssignment = dict[object, bool]
OpaqueAssignmentSequence = list[OpaqueAssignment]


@dataclass(frozen=True)
class OpaqueAssignmentScenario:
    """Per-task opaque assignments for one joint interleaving check.

    A task with a ``while_true_loop()`` receives an assignment sequence, one
    dictionary per modeled round. Other tasks receive ``default_assignment``,
    the final (break) round of the first while-loop task, so they observe the
    exit-round values of shared opaque keys.
    """

    default_assignment: OpaqueAssignment
    task_assignments: dict[int, OpaqueAssignmentSequence]
    task_names: dict[int, str] = field(default_factory=dict)

    def for_task(self, task: Task) -> OpaqueAssignment | OpaqueAssignmentSequence:
        return self.task_assignments.get(id(task), self.default_assignment)


# Stages that make progress in a retry round. Auxiliary work, such as an exit
# probe, does not: a round that only polls can spin forever.
_WHILE_RETRY_PROGRESS_STAGES = frozenset(
    {
        ScheduleStage.ProducerTryAcquire,
        ScheduleStage.ProducerAcquire,
        ScheduleStage.ProducerCommit,
        ScheduleStage.ProducerWork,
        ScheduleStage.ConsumerTryWait,
        ScheduleStage.ConsumerWait,
        ScheduleStage.ConsumerRelease,
        ScheduleStage.ConsumerWork,
    }
)


def format_opaque_assignment(assignment: object) -> str:
    """Render an assignment, a while-loop sequence, or a joint scenario."""
    if isinstance(assignment, OpaqueAssignmentScenario):
        parts = [f"default={format_opaque_assignment(assignment.default_assignment)}"]
        for task_id, sequence in assignment.task_assignments.items():
            name = assignment.task_names.get(task_id, str(task_id))
            parts.append(f"{name}: {format_opaque_assignment(sequence)}")
        return "; ".join(parts)
    if isinstance(assignment, list):
        return " -> ".join(format_opaque_assignment(item) for item in assignment)
    if not assignment:
        return "{}"
    return ", ".join(f"{key}={value}" for key, value in assignment.items())


def expand_task(
    task: Task,
    *,
    skipped_tile: bool = False,
    opaque_assignment: (
        OpaqueAssignment | OpaqueAssignmentSequence | OpaqueAssignmentScenario | None
    ) = None,
    num_tiles: int = 1,
    dynamic_domain_fallback: int = 1,
    representative_domain: bool = False,
) -> list[FlatScheduleOp]:
    """Flatten one task schedule into the operation order the checker explores.

    ``opaque_assignment`` is a dictionary, a while-loop sequence, or an
    :class:`OpaqueAssignmentScenario`. A sequence unrolls the task's
    ``while_true_loop()`` once per round; guards outside that loop observe its
    final (break) round. Init/advance hooks are not emitted: they compute the
    cursor and do not touch pipeline credits.
    """
    validate_break_placement(task.schedule.body)
    if isinstance(opaque_assignment, OpaqueAssignmentScenario):
        opaque_assignment = opaque_assignment.for_task(task)
    while_sequence = None
    if isinstance(opaque_assignment, list):
        while_sequence = opaque_assignment
        opaque_assignment = opaque_assignment[-1] if opaque_assignment else {}
    result = []
    domain_entry = itertools.count()

    def emit(nodes, assignment, iteration=None, count=None, phase="O", skipped=False):
        for node in nodes:
            if isinstance(node, BreakLoop):
                return True
            if isinstance(node, Step):
                # Static work-queue stages only update task-local scheduler
                # state; unlike a CLC queue, they do not synchronize tasks.
                if (
                    isinstance(node.memory_resource, WorkQueue)
                    and node.memory_resource.pipeline_config is None
                ):
                    continue
                result.append(
                    FlatScheduleOp(
                        node.memory_resource,
                        node.schedule_stage,
                        node.unique_id,
                        phase,
                    )
                )
            elif isinstance(node, DomainLoop):
                if skipped:
                    continue
                count = _domain_trip_count(
                    node, task, dynamic_domain_fallback, representative_domain
                )
                next(domain_entry)
                for index in range(count):
                    if emit(node.body, assignment, index, count, "", skipped):
                        break
            elif isinstance(node, WhileTrueLoop):
                if skipped:
                    continue
                emit_while(node, assignment)
            elif isinstance(node, WorkTileLoop):
                for _ in range(num_tiles):
                    emit(node.body, assignment, iteration, count, phase, skipped_tile)
            elif isinstance(node, ConditionalBlock):
                guard = node.condition
                if guard is SKIPPABLE:
                    active = not skipped
                else:
                    active = guard_fires(
                        guard,
                        iteration or 0,
                        count or 0,
                        opaque_assignment=assignment,
                    )
                if active:
                    guard_phase = _guard_phase(guard, phase)
                    if emit(node.body, assignment, iteration, count, guard_phase, skipped):
                        return True
        return False

    def emit_while(node, assignment):
        # Without a sequence, the round's assignment is the plain dictionary;
        # the structural round checks apply only to modeled sequences.
        sequence = while_sequence if while_sequence is not None else [assignment or {}]
        if not sequence:
            raise ValueError(
                f"Task {task.name!r}: while_true_loop() received an empty "
                "assignment sequence"
            )
        for index, round_assignment in enumerate(sequence):
            start = len(result)
            broke = emit(node.body, round_assignment, index, len(sequence), "")
            if while_sequence is None:
                if broke:
                    break
                continue
            where = f"while_true_loop() round {index + 1} of {len(sequence)}"
            if index == len(sequence) - 1:
                if not broke:
                    raise ValueError(
                        f"Task {task.name!r}: {where} does not reach break_loop()"
                    )
            elif broke:
                raise ValueError(
                    f"Task {task.name!r}: retry {where} reaches break_loop()"
                )
            elif not any(
                op.schedule_stage in _WHILE_RETRY_PROGRESS_STAGES
                for op in result[start:]
            ):
                raise ValueError(
                    f"Task {task.name!r}: retry {where} has no blocking or "
                    "state-changing pipeline operation"
                )

    emit(task.schedule.body, opaque_assignment)
    return result


def collect_opaque_keys(tasks: list[Task]) -> list[object]:
    """Return distinct opaque leaf keys of every guard, in first-seen order.

    Values used only as routed data are not verifier keys.
    """
    result = []
    for task in tasks:
        for node in _iter_nodes(task.schedule.body):
            if isinstance(node, ConditionalBlock):
                for leaf in iter_leaves(node.condition):
                    if isinstance(leaf, OpaqueCondition) and leaf.key not in result:
                        result.append(leaf.key)
    return result


GuardObservation = tuple[BlockGuard, int, int]


def collect_guard_observations(
    tasks: list[Task], *, representative_domain: bool = False
) -> list[GuardObservation]:
    """Collect each guard at every enclosing domain iteration it can observe.

    Each observation is ``(guard, iter_idx, num_iters)``, using the trip counts
    :func:`expand_task` models. Observations that agree on every iteration leaf
    are kept once, since they evaluate identically under any assignment.
    """
    fallback = dynamic_domain_fallback_for(tasks)
    observations = []
    seen = set()

    def observe(guard, contexts):
        iteration_leaves = [
            leaf for leaf in iter_leaves(guard) if isinstance(leaf, IterationPredicate)
        ]
        for index, count in contexts:
            signature = (
                id(guard),
                tuple(leaf.fires(index, count) for leaf in iteration_leaves),
            )
            if signature not in seen:
                seen.add(signature)
                observations.append((guard, index, count))

    def walk(task, nodes, contexts):
        for node in nodes:
            if isinstance(node, (Step, BreakLoop)):
                continue
            if isinstance(node, DomainLoop):
                count = _domain_trip_count(node, task, fallback, representative_domain)
                walk(task, node.body, tuple((index, count) for index in range(count)))
                continue
            if isinstance(node, ConditionalBlock) and node.condition is not SKIPPABLE:
                observe(node.condition, contexts)
            walk(task, node.body, contexts)

    for task in tasks:
        walk(task, task.schedule.body, ((0, 0),))
    return observations


def find_while_true_loop(task: Task) -> WhileTrueLoop | None:
    """Return the task's ``while_true_loop()``; the builder allows at most one."""
    return next(
        (
            node
            for node in _iter_nodes(task.schedule.body)
            if isinstance(node, WhileTrueLoop)
        ),
        None,
    )


def _all_opaque_assignments(
    keys: list[object],
    observations: list[GuardObservation] | None = None,
) -> list[OpaqueAssignment]:
    """Return one assignment per observably distinct control-flow class.

    Every key maps to ``True`` or ``False``. With ``observations``, two
    assignments are equivalent when every observed guard evaluates the same
    under both; the first of each class represents it. For example, a schedule
    that observes only ``a & b`` keeps two classes, not four.
    """
    assignments = [
        dict(zip(keys, values))
        for values in itertools.product((False, True), repeat=len(keys))
    ]
    if not observations:
        return assignments
    distinct = {}
    for assignment in assignments:
        signature = tuple(
            guard_fires(guard, index, count, opaque_assignment=assignment)
            for guard, index, count in observations
        )
        distinct.setdefault(signature, assignment)
    return list(distinct.values())


def _clause_holds(clause: tuple, assignment: OpaqueAssignment) -> bool:
    """An empty clause is an unconditional break and always holds."""
    return all(
        guard_fires(guard, 0, 0, opaque_assignment=assignment) for guard in clause
    )


def _fire_count_hints(loop: WhileTrueLoop) -> list[tuple[tuple, int]]:
    """Collect ``verify_num_fires`` assertions with their effective guards.

    A nested block runs only when every enclosing runtime guard holds, so its
    effective guard is the conjunction of the whole chain.
    """
    hints = []

    def walk(nodes, chain):
        for node in nodes:
            if isinstance(node, ConditionalBlock):
                inner = chain
                if node.condition is not SKIPPABLE:
                    inner = chain + (node.condition,)
                if node.verify_num_fires is not None:
                    hints.append((inner, node.verify_num_fires))
                walk(node.body, inner)

    walk(loop.body, ())
    return hints


def _while_task_sequences(
    task: Task,
    loop: WhileTrueLoop,
    keys: list[object],
    observations: list[GuardObservation] | None = None,
) -> list[OpaqueAssignmentSequence]:
    """Enumerate ``retry^(N-1) + break`` round histories for one task.

    A break assignment satisfies at least one exit clause; a retry assignment
    satisfies none. ``verify_num_fires`` assertions discard histories in which
    a block would run a different number of times.
    """
    clauses = while_loop_break_clauses(loop)
    assignments = _all_opaque_assignments(keys, observations)
    breaks = [a for a in assignments if any(_clause_holds(c, a) for c in clauses)]
    retries = [a for a in assignments if not any(_clause_holds(c, a) for c in clauses)]
    if not breaks:
        raise ValueError(
            f"Task {task.name!r}: while_true_loop() has no opaque assignment that "
            "satisfies a break_loop() clause"
        )
    hints = _fire_count_hints(loop)
    sequences = []
    for count in loop.verify_num_iters:
        if count > 1 and not retries:
            raise ValueError(
                f"Task {task.name!r}: while_true_loop() verify_num_iters={count} "
                "requires a retry assignment, but every opaque assignment "
                "satisfies a break_loop() clause"
            )
        for history in itertools.product(retries, repeat=count - 1):
            for final in breaks:
                sequence = [*history, final]
                if all(
                    sum(_clause_holds(guards, a) for a in sequence) == declared
                    for guards, declared in hints
                ):
                    sequences.append(sequence)
    if not sequences:
        widest = max(loop.verify_num_iters)
        details = []
        for guards, declared in hints:
            retry_fires = [_clause_holds(guards, a) for a in retries]
            break_fires = [_clause_holds(guards, a) for a in breaks]
            low = (widest - 1) * bool(retries and all(retry_fires)) + all(break_fires)
            high = (widest - 1) * any(retry_fires) + any(break_fires)
            reach = f"exactly {low}" if low == high else f"between {low} and {high}"
            rendered = " and ".join(_format_guard(guard) for guard in guards)
            details.append(
                f"{rendered} declares verify_num_fires={declared} but fires on "
                f"{reach} of {widest} modeled rounds"
            )
        raise ValueError(
            f"Task {task.name!r}: no while_true_loop() round history satisfies "
            f"verify_num_fires over verify_num_iters={loop.verify_num_iters}: "
            f"{'; '.join(details)}. Raise verify_num_iters to model more rounds, "
            "or correct the assertion"
        )
    return sequences


def opaque_assignment_sequences(
    tasks: list[Task],
    keys: list[object],
    *,
    representative_domain: bool = False,
) -> list[OpaqueAssignment | OpaqueAssignmentScenario]:
    """Enumerate the opaque inputs for every exhaustive interleaving check.

    Without a ``while_true_loop()`` this is one boolean assignment of ``keys``
    per observably distinct control-flow class. Otherwise each result is a
    scenario combining one round history per while-loop task.
    """
    observations = collect_guard_observations(
        tasks, representative_domain=representative_domain
    )
    while_tasks = [
        (task, loop)
        for task in tasks
        if (loop := find_while_true_loop(task)) is not None
    ]
    if not while_tasks:
        return _all_opaque_assignments(keys, observations)
    names = {id(task): task.name for task, _ in while_tasks}
    return [
        OpaqueAssignmentScenario(
            default_assignment=histories[0][-1],
            task_assignments={
                id(task): history
                for (task, _), history in zip(while_tasks, histories)
            },
            task_names=names,
        )
        for histories in itertools.product(
            *(
                _while_task_sequences(task, loop, keys, observations)
                for task, loop in while_tasks
            )
        )
    ]


def dynamic_domain_fallback_for(tasks: list[Task]) -> int:
    starts = [
        node.start
        for task in tasks
        for node in _iter_nodes(task.schedule.body)
        if isinstance(node, DomainLoop) and type(node.start) is int
    ]
    return max(starts, default=0) + 1


def _explode_sync_work(schedules: list[list[FlatScheduleOp]]) -> list[list[FlatScheduleOp]]:
    """Expand non-pipelined work into two logical synchronization phases."""
    result = []
    for schedule in schedules:
        expanded = []
        for operation in schedule:
            resource = operation.resource
            stage = operation.schedule_stage
            is_nonpipeline = not resource.is_barrier and resource.pipeline_config is None
            if is_nonpipeline and stage is ScheduleStage.ConsumerWork:
                expanded.append(replace(operation, sync_phase="cons_write_phase"))
                expanded.append(replace(operation, sync_phase="cons_read_phase"))
            elif is_nonpipeline and stage is ScheduleStage.ProducerWork:
                expanded.extend((operation, operation))
            else:
                expanded.append(operation)
        result.append(expanded)
    return result


@dataclass(slots=True)
class DeadlockInfo:
    cursors: tuple[int, ...]
    blocked_tasks: tuple[str, ...]

    def format_lines(self) -> list[str]:
        return [
            f"deadlock witness at cursors {self.cursors}: "
            f"blocked tasks {', '.join(self.blocked_tasks)}"
        ]


@dataclass(slots=True)
class RaceInfo:
    cursors: tuple[int, ...]
    writer_task: str
    writer_resource: str
    victim_task: str
    victim_resource: str
    victim_access: str
    overlap_desc: str


@dataclass(slots=True)
class PdlOrderInfo:
    cursors: tuple[int, ...]
    launch_task: str
    launch_resource: str


@dataclass(slots=True)
class CheckResult:
    deadlock_states: list[DeadlockInfo]
    race_states: list[RaceInfo]
    states_explored: int
    complete_count: int
    is_safe: bool
    pdl_order_states: list[PdlOrderInfo] = field(default_factory=list)
    hit_state_limit: bool = False


@dataclass(slots=True, frozen=True)
class _TraceNode:
    parent: "_TraceNode | None"
    task_index: int
    operation: str


def _trace_items(node: _TraceNode | None) -> list[tuple[int, str]]:
    items = []
    while node is not None:
        items.append((node.task_index, node.operation))
        node = node.parent
    items.reverse()
    return items


class _VerboseReporter:
    """Render the expanded schedules and representative BFS witnesses."""

    def __init__(self, tasks: list[Task]) -> None:
        self.tasks = tasks

    def print_header(
        self,
        schedules: list[list[FlatScheduleOp]],
        *,
        skipped_tile: bool,
    ) -> None:
        print("=" * 72)
        print("Exhaustive interleaving checker - BFS exploration")
        if skipped_tile:
            print("Variant: skipped-tile execution")
        print("=" * 72)
        for task_index, task in enumerate(self.tasks):
            schedule = ", ".join(
                f"{op.schedule_stage.name}({op.resource.name})"
                for op in schedules[task_index]
            )
            print(f"  Task {task_index} ({task.name}): [{schedule}]")
        print("-" * 72)

    def print_timeline(
        self,
        trace: _TraceNode | None,
        *,
        suffix_label: str = "",
    ) -> None:
        path = _trace_items(trace)
        if not path:
            print("  timeline: (initial state)")
            if suffix_label:
                print(f"  {suffix_label}")
            return

        column_width = (
            max(
                max((len(operation) for _, operation in path), default=0),
                max((len(task.name) for task in self.tasks), default=0),
                5,
            )
            + 2
        )
        header = "    t  "
        separator = "    -  "
        for task in self.tasks:
            header += task.name.ljust(column_width)
            separator += ("-" * len(task.name)).ljust(column_width)
        print(f"  timeline ({len(path)} steps):")
        print(header)
        print(separator)
        for time, (active_task, operation) in enumerate(path):
            row = f"    {time:<3d}"
            for task_index in range(len(self.tasks)):
                cell = operation if task_index == active_task else "."
                row += cell.ljust(column_width)
            print(row)
        if suffix_label:
            print(f"  {suffix_label}")

    def format_held(self, held: tuple, resources: dict[int, object]) -> str:
        if not held:
            return "{}"
        entries = []
        for task_index, resource_id, access in held:
            entries.append(
                f"{resources[resource_id].name}:{access}@"
                f"{self.tasks[task_index].name}"
            )
        return "{" + ", ".join(entries) + "}"

    def print_complete(
        self,
        state_number: int,
        cursors: tuple[int, ...],
        trace: _TraceNode | None,
    ) -> None:
        print(f"State #{state_number}  cursors={cursors}  COMPLETE")
        self.print_timeline(trace)
        print()

    def print_deadlock(
        self,
        state_number: int,
        cursors: tuple[int, ...],
        held: tuple,
        resources: dict[int, object],
        trace: _TraceNode | None,
        blocked_tasks: tuple[str, ...],
    ) -> None:
        print(
            f"State #{state_number}  cursors={cursors}  "
            f"held={self.format_held(held, resources)}"
        )
        self.print_timeline(
            trace,
            suffix_label=f"*** DEADLOCK: blocked tasks {', '.join(blocked_tasks)}",
        )
        print()

    def print_race(
        self,
        state_number: int,
        cursors: tuple[int, ...],
        held: tuple,
        resources: dict[int, object],
        trace: _TraceNode | None,
        race: RaceInfo,
    ) -> None:
        print(
            f"State #{state_number}  cursors={cursors}  "
            f"held={self.format_held(held, resources)}"
        )
        self.print_timeline(
            trace,
            suffix_label=(
                f"*** RACE: {race.writer_task} writes {race.writer_resource} vs "
                f"{race.victim_task} {race.victim_access} "
                f"{race.victim_resource} ({race.overlap_desc})"
            ),
        )
        print()

    @staticmethod
    def print_summary(result: CheckResult, max_states: int) -> None:
        print("=" * 72)
        print(
            f"BFS complete: {result.states_explored} states explored, "
            f"{result.complete_count} complete, "
            f"{len(result.deadlock_states)} deadlock(s), "
            f"{len(result.race_states)} race(s), "
            f"{len(result.pdl_order_states)} PDL order violation(s)"
        )
        if result.hit_state_limit:
            print(
                f"WARNING: exhaustive checker hit max_states={max_states} "
                "before completing the search; no concrete issue was found."
            )
        print(f"Result: {'SAFE' if result.is_safe else 'UNSAFE'}")
        print("=" * 72)


def _physical_ranges(resource: object) -> list[tuple[str, int, int]]:
    return getattr(resource, "physical_ranges", lambda: [])()


def _overlap(left: object, right: object) -> str | None:
    for left_space, left_start, left_end in _physical_ranges(left):
        for right_space, right_start, right_end in _physical_ranges(right):
            if left_space == right_space and max(left_start, right_start) < min(
                left_end, right_end
            ):
                return (
                    f"{left_space}[{max(left_start, right_start)}:"
                    f"{min(left_end, right_end)}]"
                )
    return None


def check_all_interleavings(
    tasks: list[Task],
    alias_map=None,
    prod_alias_map=None,
    cons_alias_map=None,
    overlap_descs=None,
    max_states: int = 1_000_000,
    num_tiles: int = 1,
    skipped_tile: bool = False,
    assume_pdl_wait_completed: bool = False,
    verbose: bool = False,
    opaque_assignment: (
        OpaqueAssignment | OpaqueAssignmentSequence | OpaqueAssignmentScenario | None
    ) = None,
    early_exit: bool = True,
    cursor_only_visited: bool = True,
    representative_domain: bool = False,
) -> CheckResult:
    del alias_map, prod_alias_map, cons_alias_map, overlap_descs
    del assume_pdl_wait_completed
    if not tasks:
        result = CheckResult([], [], 0, 1, True)
        if verbose:
            _VerboseReporter.print_summary(result, max_states)
        return result
    fallback = dynamic_domain_fallback_for(tasks)
    schedules = [
        expand_task(
            task,
            skipped_tile=skipped_tile,
            opaque_assignment=opaque_assignment,
            num_tiles=num_tiles,
            dynamic_domain_fallback=fallback,
            representative_domain=representative_domain,
        )
        for task in tasks
    ]
    schedules = _explode_sync_work(schedules)
    resources = {
        id(op.resource): op.resource for schedule in schedules for op in schedule
    }
    reporter = _VerboseReporter(tasks) if verbose else None
    if reporter is not None:
        reporter.print_header(schedules, skipped_tile=skipped_tile)
    initial_prod = tuple(
        (
            id(resource),
            resource.pipeline_config.num_stages
            if resource.pipeline_config is not None
            else 1,
        )
        for resource in resources.values()
    )
    initial_cons = tuple((rid, 0) for rid in resources)
    clc_consumer_counts = {
        rid: sum(
            any(
                id(op.resource) == rid
                and op.schedule_stage is ScheduleStage.ConsumerWait
                for op in schedule
            )
            for schedule in schedules
        )
        for rid, resource in resources.items()
        if resource.pipeline_config is not None
        and resource.pipeline_config.pipeline_type is PipelineType.ClcFetchAsync
    }
    initial_releases = tuple((rid, 0) for rid in clc_consumer_counts)
    queue = deque(
        [
            (
                (0,) * len(tasks),
                initial_prod,
                initial_cons,
                initial_releases,
                (),
                None,
            )
        ]
    )
    visited = set()
    deadlocks = []
    races = []
    explored = complete = 0

    while queue and explored < max_states:
        cursors, frozen_prod, frozen_cons, frozen_releases, held, trace = (
            queue.popleft()
        )
        key = (
            cursors
            if cursor_only_visited
            else (cursors, frozen_prod, frozen_cons, frozen_releases, held)
        )
        if key in visited:
            continue
        visited.add(key)
        explored += 1
        if all(cursors[i] == len(schedules[i]) for i in range(len(tasks))):
            complete += 1
            if reporter is not None:
                reporter.print_complete(explored, cursors, trace)
            continue
        prod, cons = dict(frozen_prod), dict(frozen_cons)
        releases = dict(frozen_releases)
        enabled = 0
        for task_index, task in enumerate(tasks):
            cursor = cursors[task_index]
            if cursor >= len(schedules[task_index]):
                continue
            op = schedules[task_index][cursor]
            rid = id(op.resource)
            stage = op.schedule_stage
            if stage is ScheduleStage.ProducerAcquire and prod[rid] <= 0:
                continue
            if stage is ScheduleStage.ConsumerWait and cons[rid] <= 0:
                continue
            enabled += 1
            next_prod, next_cons = dict(prod), dict(cons)
            next_releases = dict(releases)
            next_held = list(held)
            if stage is ScheduleStage.ProducerAcquire:
                next_prod[rid] -= 1
                next_held.append((task_index, rid, "write"))
            elif stage is ScheduleStage.ProducerCommit:
                next_cons[rid] += clc_consumer_counts.get(rid, 1)
                next_held = [h for h in next_held if h != (task_index, rid, "write")]
            elif stage is ScheduleStage.ConsumerWait:
                next_cons[rid] -= 1
                next_held.append((task_index, rid, "read"))
            elif stage is ScheduleStage.ConsumerRelease:
                if rid in clc_consumer_counts:
                    next_releases[rid] += 1
                    if next_releases[rid] == clc_consumer_counts[rid]:
                        next_prod[rid] += 1
                        next_releases[rid] = 0
                else:
                    next_prod[rid] += 1
                next_held = [h for h in next_held if h != (task_index, rid, "read")]
            access = None
            if stage is ScheduleStage.ProducerWork:
                access = "write"
            elif stage is ScheduleStage.ConsumerWork:
                access = (
                    "write"
                    if op.sync_phase == "cons_write_phase"
                    else "read"
                )
            if access:
                for owner, held_rid, held_access in held:
                    if (
                        owner == task_index
                        or rid == held_rid
                        or (access == held_access == "read")
                    ):
                        continue
                    description = _overlap(op.resource, resources[held_rid])
                    if description:
                        races.append(
                            RaceInfo(
                                cursors,
                                task.name if access == "write" else tasks[owner].name,
                                op.resource.name
                                if access == "write"
                                else resources[held_rid].name,
                                tasks[owner].name if access == "write" else task.name,
                                resources[held_rid].name
                                if access == "write"
                                else op.resource.name,
                                held_access,
                                description,
                            )
                        )
                        if early_exit:
                            result = CheckResult(
                                deadlocks, races, explored, complete, False
                            )
                            if reporter is not None:
                                reporter.print_race(
                                    explored,
                                    cursors,
                                    held,
                                    resources,
                                    trace,
                                    races[-1],
                                )
                                reporter.print_summary(result, max_states)
                            return result
            next_cursors = list(cursors)
            next_cursors[task_index] += 1
            next_trace = (
                _TraceNode(
                    trace,
                    task_index,
                    f"{stage.name}({op.resource.name})",
                )
                if verbose
                else None
            )
            queue.append(
                (
                    tuple(next_cursors),
                    tuple(sorted(next_prod.items())),
                    tuple(sorted(next_cons.items())),
                    tuple(sorted(next_releases.items())),
                    tuple(sorted(next_held)),
                    next_trace,
                )
            )
        if enabled == 0:
            blocked = tuple(
                tasks[i].name
                for i, cursor in enumerate(cursors)
                if cursor < len(schedules[i])
            )
            deadlocks.append(DeadlockInfo(cursors, blocked))
            if reporter is not None:
                reporter.print_deadlock(
                    explored,
                    cursors,
                    held,
                    resources,
                    trace,
                    blocked,
                )
            if early_exit:
                break
    safe = not deadlocks and not races and bool(complete)
    result = CheckResult(
        deadlocks,
        races,
        explored,
        complete,
        safe,
        hit_state_limit=bool(queue and explored >= max_states),
    )
    if reporter is not None:
        reporter.print_summary(result, max_states)
    return result
