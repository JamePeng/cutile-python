# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Task definitions for the SM120 FMHA context kernel."""

import task_scheduling as ts


def _captured_loop_bounds(
    task_kwargs: dict, default_start: int = 0
) -> tuple[object, object, object]:
    """Pop loop-bound kwargs and return ``(start, end, step)``."""
    loop_start = task_kwargs.pop("domain_start", default_start)
    loop_step = task_kwargs.pop("step", 1)
    loop_end = task_kwargs.pop("domain", None)
    if loop_end is None:
        loop_end = task_kwargs.get("num_kv_tiles")
    if loop_end is None:
        raise ValueError("FMHA tasks require a domain bound")
    return loop_start, loop_end, loop_step


def create_load_task(
    cfg,
    gmem_qkv,
    smem_k,
    smem_v,
    task_class=ts.Task,
    **task_kwargs,
):
    """Create the TMA load task."""
    loop_start, loop_end, loop_step = _captured_loop_bounds(task_kwargs)

    @ts.schedule
    def load_schedule(stage_info, gqkv, sk, sv):
        k_smem = sk.init_load_state(cfg.k_smem_offset, cfg.tile_elements)
        v_smem = sv.init_load_state(cfg.v_smem_offset, cfg.tile_elements)
        batch_coord, head_coord, seq_coord_q = gqkv.compute_coords(
            cfg.q_tile,
            cfg.is_causal,
            cfg.use_causal_head_fast_grid,
        )
        _ = seq_coord_q

        def loop_body():
            sk.try_acquire()
            sk.acquire()
            sk.tma_load(
                k_smem,
                batch_coord,
                head_coord,
                cfg.q_tile,
                cfg.kv_tile,
                cfg.is_causal,
                cfg.use_causal_head_fast_grid,
            )
            sk.commit()
            sv.try_acquire()
            sv.acquire()
            sv.tma_load(
                v_smem,
                batch_coord,
                head_coord,
                cfg.q_tile,
                cfg.kv_tile,
                cfg.is_causal,
                cfg.use_causal_head_fast_grid,
            )
            sv.commit()

        ts.domain_loop(
            loop_start,
            loop_end,
            loop_step,
            loop_body,
        )

    captured_schedule = load_schedule(gmem_qkv, smem_k, smem_v)
    return task_class(
        name="LoadTask",
        warp_idx=cfg.load_warp_index,
        num_warps=cfg.num_load_warps,
        schedule=captured_schedule,
        num_registers=cfg.load_regs,
        **task_kwargs,
    )


def create_compute_task(
    cfg,
    smem_k,
    smem_v,
    gmem_o,
    task_class=ts.Task,
    **task_kwargs,
):
    """Create the QK, online-softmax, PV, and epilogue task."""
    loop_start, loop_end, loop_step = _captured_loop_bounds(
        task_kwargs, default_start=1
    )
    static_attention_args = (
        cfg.head_dim,
        cfg.q_tile,
        cfg.kv_tile,
        cfg.num_compute_warps,
        cfg.tma_swizzle_chunk_elems,
        cfg.dtype == "bf16",
        cfg.is_causal,
        cfg.use_causal_head_fast_grid,
    )

    @ts.schedule
    def compute_schedule(stage_info, sk, sv, go):
        k_smem = sk.init_compute_state(cfg.tile_elements)
        v_smem = sv.init_compute_state(cfg.tile_elements, cfg.tma_copy_kv_bytes)
        accumulator, row_max, row_sum = sv.init_compute_work_tile_state(cfg.head_dim)
        q_registers = sk.load_q(
            cfg.head_dim,
            cfg.q_tile,
            cfg.num_compute_warps,
            cfg.is_causal,
            cfg.use_causal_head_fast_grid,
        )

        sk.try_wait()
        sk.wait()
        scores = sk.qk_mma(
            k_smem,
            q_registers,
            cfg.head_dim,
            cfg.kv_tile,
            cfg.tma_swizzle_chunk_elems,
            cfg.dtype == "bf16",
        )
        sk.release()
        sv.try_wait()
        sv.wait()
        accumulator, row_max, row_sum = sv.softmax_pv(
            v_smem,
            scores,
            accumulator,
            row_max,
            row_sum,
            0,
            *static_attention_args,
        )
        sv.release()

        def loop_body(accumulator, row_max, row_sum):
            sk.try_wait()
            sk.wait()
            scores = sk.qk_mma(
                k_smem,
                q_registers,
                cfg.head_dim,
                cfg.kv_tile,
                cfg.tma_swizzle_chunk_elems,
                cfg.dtype == "bf16",
            )
            sk.release()
            sv.try_wait()
            sv.wait()
            accumulator, row_max, row_sum = sv.softmax_pv_with_correction(
                v_smem,
                scores,
                accumulator,
                row_max,
                row_sum,
                *static_attention_args,
            )
            sv.release()
            return accumulator, row_max, row_sum

        accumulator, row_max, row_sum = ts.domain_loop(
            loop_start,
            loop_end,
            loop_step,
            loop_body,
            accumulator,
            row_max,
            row_sum,
        )
        go.epilogue_and_store(
            accumulator,
            row_sum,
            cfg.head_dim,
            cfg.q_tile,
            cfg.num_compute_warps,
            cfg.tile_elements,
            cfg.dtype == "bf16",
            cfg.is_causal,
            cfg.use_causal_head_fast_grid,
        )

    captured_schedule = compute_schedule(smem_k, smem_v, gmem_o)
    return task_class(
        name="ComputeTask",
        warp_idx=0,
        num_warps=cfg.num_compute_warps,
        schedule=captured_schedule,
        num_registers=cfg.compute_regs,
        **task_kwargs,
    )


def create_padding_task(cfg, task_class=ts.Task, **task_kwargs):
    """Create the no-op padding task used for register budgeting."""
    del task_class
    for key in (
        "domain",
        "domain_start",
        "step",
        "num_kv_tiles",
        "seqlen_q",
        "q_tile",
        "kv_tile",
        "num_heads_q",
        "use_head_fast_grid",
    ):
        task_kwargs.pop(key, None)

    @ts.schedule
    def padding_schedule(stage_info):
        with ts.domain_loop(0, 1, 1):
            pass

    captured_schedule = padding_schedule()
    return ts.Task(
        name="PaddingTask",
        warp_idx=cfg.padding_warp_index,
        num_warps=cfg.num_padding_warps,
        schedule=captured_schedule,
        num_registers=cfg.padding_regs,
        **task_kwargs,
    )
