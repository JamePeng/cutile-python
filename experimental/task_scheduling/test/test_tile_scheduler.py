# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import task_scheduling as ts
from task_scheduling.enums import TileSchedulerType
from task_scheduling.task import (
    DeviceStep,
    DeviceWorkQueueAdvance,
    DeviceWorkTileLoop,
)


def _work_queue_advance(device_task):
    work_tile_loop = next(
        node for node in device_task.body if isinstance(node, DeviceWorkTileLoop)
    )
    assert isinstance(work_tile_loop, DeviceWorkTileLoop)
    return next(
        node
        for node in work_tile_loop.body
        if isinstance(node, DeviceWorkQueueAdvance)
    )


def test_static_persistent_grid_is_cluster_aligned_and_occupancy_capped():
    params = ts.PersistentTileSchedulerParams(
        problem_shape_ntile_mnl=(16, 8, 1),
        cluster_shape_mnk=(2, 1, 1),
    )

    assert params.problem_shape_ncluster_mnl == (8, 8, 1)
    assert params.problem_cluster_count == 64
    assert ts.StaticPersistentTileScheduler.get_grid_shape(params, 20) == (2, 1, 20)


def test_work_tile_loop_freezes_with_static_persistent_scheduler():
    params = ts.PersistentTileSchedulerParams(
        problem_shape_ntile_mnl=(4, 2, 1),
        cluster_shape_mnk=(2, 1, 1),
    )
    queue = ts.WorkQueue(
        name="queue",
        tile_scheduler_config=(
            ts.TileSchedulerConfig.create_static_persistent_tile_scheduler_params(
                params
            )
        ),
    )

    @ts.schedule
    def persistent(wq):
        with ts.work_tile_loop(wq):
            wq.try_wait()
            wq.wait()
            wq.get_and_advance_work_tile()
            wq.release()

    task = ts.Task(0, 1, schedule=persistent(queue), name="PersistentTask")
    manager = ts.TaskManager([task], {}, verbose=False)

    device_manager = manager.to_device()
    assert len(device_manager.tasks) == 1
    work_tile_loop = next(
        node
        for node in device_manager.tasks[0].body
        if isinstance(node, DeviceWorkTileLoop)
    )
    assert work_tile_loop.skip_if is None
    assert work_tile_loop.skip_context is None
    assert not work_tile_loop.bind_skip_context
    advance = _work_queue_advance(device_manager.tasks[0])
    assert advance.tile_scheduler_type is TileSchedulerType.StaticPersistent
    assert advance.pipeline_slot == -1


def test_work_tile_loop_freezes_skip_predicate():
    params = ts.PersistentTileSchedulerParams(
        problem_shape_ntile_mnl=(4, 2, 1),
        cluster_shape_mnk=(1, 1, 1),
    )
    queue = ts.WorkQueue(
        name="queue",
        tile_scheduler_config=(
            ts.TileSchedulerConfig.create_static_persistent_tile_scheduler_params(
                params
            )
        ),
    )

    def skip_if(_queue, work_tile):
        return work_tile.tile_idx[0] >= 2

    @ts.schedule
    def persistent(wq):
        with ts.work_tile_loop(wq, skip_if=skip_if) as work_tiles:
            with work_tiles.skippable():
                wq.try_wait()
            wq.get_and_advance_work_tile()

    task = ts.Task(0, 1, schedule=persistent(queue), name="PersistentTask")
    manager = ts.TaskManager([task], {}, verbose=False)
    device_task = manager.to_device().tasks[0]
    work_tile_loop = next(
        node for node in device_task.body if isinstance(node, DeviceWorkTileLoop)
    )

    assert work_tile_loop.skip_if is skip_if
    assert work_tile_loop.skip_context is queue


def test_pdl_resources_use_their_static_work_methods():
    pdl_wait = ts.PdlWaitBarrier(name="pdl_wait")
    pdl_launch = ts.PdlLaunchBarrier(name="pdl_launch")

    @ts.schedule
    def pdl_sync(wait, launch):
        wait.wait_griddep()
        launch.launch_griddep()

    task = ts.Task(
        0,
        1,
        schedule=pdl_sync(pdl_wait, pdl_launch),
        name="PdlTask",
    )
    manager = ts.TaskManager(
        [task],
        {},
        verbose=False,
        exhaustive_deadlock_race_check=False,
    )

    device_manager = manager.to_device()
    wait_step, launch_step = device_manager.tasks[0].body
    assert isinstance(wait_step, DeviceStep)
    assert isinstance(launch_step, DeviceStep)
    assert wait_step.callback.__name__ == "wait_griddep"
    assert launch_step.callback.__name__ == "launch_griddep"


def test_clc_dynamic_grid_and_dual_role_work_queue_freeze():
    params = ts.ClcDynamicPersistentTileSchedulerParams(
        problem_shape_ntile_mnl=(15, 8, 1),
        cluster_shape_mnk=(2, 1, 1),
    )
    response = ts.SmemAllocation(
        "clc_response",
        size_bytes=32,
        alignment=16,
        count=2,
    )
    queue = ts.WorkQueue(
        name="queue",
        tile_scheduler_config=(
            ts.TileSchedulerConfig.create_clc_dynamic_persistent_tile_scheduler_params(
                params,
                response,
            )
        ),
        pipeline_config=ts.PipelineConfig.create_clc_fetch_async_pipeline_cfg(
            num_stages=2,
            num_bytes=16,
            producer_group=ts.CooperativeGroup(1),
            consumer_group=ts.CooperativeGroup(64),
            cta_layout_vmnk=(2, 1, 1, 1),
            producer_signaling_threads=ts.SignalingThreads.CtaLeader,
            consumer_signaling_threads=ts.SignalingThreads.All,
        ),
        smem_requirements=[response],
    )
    pdl_wait = ts.PdlWaitBarrier(name="pdl_wait")

    @ts.schedule
    def scheduler(wait, wq):
        wait.wait_griddep()
        with ts.work_tile_loop(wq):
            wq.try_acquire()
            wq.acquire()
            wq.fetch_work_tile()
            wq.commit()
            wq.try_wait()
            wq.wait()
            wq.get_and_advance_work_tile()
            wq.release()

    task = ts.Task(
        0,
        1,
        schedule=scheduler(pdl_wait, queue),
        name="SchedulerTask",
    )
    smem_allocator = ts.SmemAllocator(default_add_barriers=False)
    smem_allocator.add_resource(queue)
    smem_allocator.compute_layout()
    barrier_allocator = ts.BarrierAllocator()
    barrier_allocator.add_resource(queue)
    barrier_allocator.compute_layout()
    manager = ts.TaskManager(
        [task],
        {},
        smem_allocator=smem_allocator,
        barrier_allocator=barrier_allocator,
        verbose=False,
    )

    assert ts.ClcDynamicPersistentTileScheduler.get_grid_shape(params) == (
        16,
        8,
        1,
    )
    device_manager = manager.to_device()
    device_task = device_manager.tasks[0]
    wait_step = device_task.body[0]
    assert isinstance(wait_step, DeviceStep)
    assert wait_step.callback.__name__ == "wait_griddep"
    assert tuple(state.phase for state in device_task.initial_pipeline_states) == (
        0,
        1,
    )
    advance = _work_queue_advance(device_task)
    assert advance.tile_scheduler_type is TileSchedulerType.ClcDynamicPersistent
    assert advance.pipeline_slot == 0
