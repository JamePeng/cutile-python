# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Captured schedules for SM100 paired D128 FMHA.

Direct and persistent call signatures are captured separately, so each task
retains two thin wrappers around one shared schedule body.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field

import cuda.lang as cl
import task_scheduling as ts

try:
    from .fmha_resources import FmhaConfig
    from .stage import FmhaStage
except ImportError:
    from fmha_resources import FmhaConfig
    from stage import FmhaStage


@dataclass(frozen=True)
class _ResolvedPackedContextWorkQueue:
    """Device state used by the packed-Q work-tile skip predicate."""

    cfg: FmhaConfig
    cum_seqlen_q: object


@dataclass(frozen=True)
class _DevicePackedContextWorkQueue:
    """Static queue metadata bound to the live packed-Q offsets at launch."""

    cfg: FmhaConfig

    def bind_inputs(self, tasks_inputs):
        return _ResolvedPackedContextWorkQueue(
            self.cfg,
            tasks_inputs.cum_seqlen_q,
        )


@dataclass(kw_only=True, eq=False)
class PackedContextWorkQueue(ts.WorkQueue):
    """Persistent queue that skips Q tiles outside a live packed request."""

    cfg: FmhaConfig = field(init=False, default=None)

    def __init__(self, cfg: FmhaConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg

    def _freeze_skip_context(self):
        return _DevicePackedContextWorkQueue(self.cfg)

    def skip_work_tile_if(self, work_tile):
        """Skip a scheduler tile whose first Q row is outside its request."""
        seq_idx, _, batch_idx = self.cfg.work_tile_coord_indices
        seq_coord = cl.int32(work_tile.tile_idx[seq_idx])
        if self.cfg.uses_causal_reversed_head_batch_seq_tile_order:
            seq_coord = cl.int32(self.cfg.num_seq_tiles) - seq_coord - cl.int32(1)
        batch_coord = cl.int32(work_tile.tile_idx[batch_idx])
        q_begin = cl.int32(self.cum_seqlen_q[batch_coord])
        q_end = cl.int32(self.cum_seqlen_q[batch_coord + cl.int32(1)])
        seqlen_q = q_end - q_begin
        return seq_coord * cl.int32(self.cfg.cta_tiler[0]) >= seqlen_q


def _persistent_tail(work_queue):
    work_queue.wait()
    work_queue.get_and_advance_work_tile()
    work_queue.release()


def _packed_context_skip_predicate(work_queue):
    """Select the live-Q skip predicate before capture creates proxies."""
    if isinstance(work_queue, PackedContextWorkQueue):
        return PackedContextWorkQueue.skip_work_tile_if
    return None


@contextmanager
def _work_tile_schedule_loop(work_queue, *, skip_if=None):
    """Run once for a direct CTA or once per persistent work-queue tile."""
    if work_queue is None:
        yield
        return
    if skip_if is not None:
        with ts.work_tile_loop(work_queue, skip_if=skip_if) as work_tiles:
            with work_tiles.skippable():
                yield
            _persistent_tail(work_queue)
    else:
        with ts.work_tile_loop(work_queue):
            yield
            _persistent_tail(work_queue)


def _src_resources(*resources, work_queue):
    """Append the scheduler dependency only for persistent schedules."""
    src = list(resources)
    if work_queue is not None:
        src.append(work_queue)
    return src


def _captured_loop_bounds(task_class, task_kwargs):
    loop_start = task_kwargs.pop("domain_start", 0)
    loop_step = task_kwargs.pop("step", 1)
    loop_end = task_kwargs.pop("domain", None)
    if loop_end is None:
        loop_end = task_class.get_domain
    return loop_start, loop_end, loop_step


def create_load_task(
    gmem_qkv,
    smem_q,
    smem_kv,
    work_queue,
    *,
    fmha_config: FmhaConfig,
    q_smem_offset,
    kv_smem_offset,
    task_class=ts.Task,
    **task_kwargs,
):
    """Build the load schedule: Q0, Ki, Q1, Vi."""

    loop_start, loop_end, loop_step = _captured_loop_bounds(
        task_class, task_kwargs
    )
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)

    def load_schedule_body(gqkv, sq, skv, wq):
        sQ_array = sq.init_load_state(
            q_smem_offset,
            fmha_config=fmha_config,
        )
        sK_array = skv.init_load_state(
            kv_smem_offset,
            fmha_config=fmha_config,
        )
        with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
            (
                seq_coord_q,
                head_coord,
                kv_head_coord,
                batch_coord,
                cuseqlen_q,
                cuseqlen_k,
                seqlen_q,
                _seqlen_k,
                kv_tile_start,
            ) = gqkv.compute_coords(fmha_config=fmha_config)

            def load_body():
                with ts.first_iter():
                    sq.acquire()
                    sq.tma_load(
                        sQ_array,
                        seq_coord_q,
                        head_coord,
                        batch_coord,
                        cuseqlen_q,
                        seqlen_q,
                        0,
                        fmha_config=fmha_config,
                    )
                    sq.commit()

                skv.try_acquire()
                skv.acquire()
                skv.k_load(
                    sK_array,
                    kv_head_coord,
                    batch_coord,
                    cuseqlen_k,
                    kv_tile_start,
                    fmha_config=fmha_config,
                )
                skv.commit()

                with ts.first_iter():
                    sq.acquire()
                    sq.tma_load(
                        sQ_array,
                        seq_coord_q,
                        head_coord,
                        batch_coord,
                        cuseqlen_q,
                        seqlen_q,
                        1,
                        fmha_config=fmha_config,
                    )
                    sq.commit()

                skv.try_acquire()
                skv.acquire()
                skv.v_load(
                    sK_array,
                    kv_head_coord,
                    batch_coord,
                    cuseqlen_k,
                    kv_tile_start,
                    fmha_config=fmha_config,
                )
                skv.commit()

            ts.domain_loop(loop_start, loop_end, loop_step, load_body)

    @ts.schedule
    def persistent_load_schedule(stage_info, gqkv, sq, skv, wq):
        load_schedule_body(gqkv, sq, skv, wq)

    @ts.schedule
    def direct_load_schedule(stage_info, gqkv, sq, skv):
        load_schedule_body(gqkv, sq, skv, None)

    schedule = (
        persistent_load_schedule(gmem_qkv, smem_q, smem_kv, work_queue)
        if work_queue is not None
        else direct_load_schedule(gmem_qkv, smem_q, smem_kv)
    )

    return task_class(
        name="LoadTask",
        warp_idx=fmha_config.load_warp_id,
        num_warps=1,
        schedule=schedule,
        num_registers=fmha_config.num_regs_other,
        **task_kwargs,
    )


def create_mma_task(
    smem_q,
    smem_kv,
    tmem_sp0,
    tmem_sp1,
    tmem_o,
    work_queue,
    *,
    fmha_config: FmhaConfig,
    q_smem_offset,
    kv_smem_offset,
    task_class=ts.Task,
    **task_kwargs,
):
    """Build the peeled HEAD / N-1 LOOP / TAIL MMA schedule."""

    loop_start, loop_end, loop_step = _captured_loop_bounds(
        task_class, task_kwargs
    )
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)

    def mma_schedule_body(sq, skv, sp0, sp1, to, wq):
        sQ_array = sq.init_descriptor_state(
            q_smem_offset,
            fmha_config=fmha_config,
        )
        sK_array = skv.init_descriptor_state(
            kv_smem_offset,
            fmha_config=fmha_config,
        )
        with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
            # HEAD: QK0(K0), QK1(K0), then PV0(V0).
            sq.wait()
            desc_q0_base = sq.q_desc(
                sQ_array,
                0,
                fmha_config=fmha_config,
            )
            skv.wait()
            desc_k_base = skv.k_desc(
                sK_array,
                fmha_config=fmha_config,
            )
            sp0.acquire()
            sp0.qk_mma(
                desc_q0_base,
                desc_k_base,
                FmhaStage.Head,
                0,
                fmha_config=fmha_config,
            )
            sp0.commit()

            sq.wait()
            desc_q1_base = sq.q_desc(
                sQ_array,
                1,
                fmha_config=fmha_config,
            )
            sp1.acquire()
            sp1.qk_mma(
                desc_q1_base,
                desc_k_base,
                FmhaStage.Head,
                1,
                fmha_config=fmha_config,
            )
            sp1.commit()
            skv.release()

            skv.wait()
            desc_v_base = skv.v_desc(
                sK_array,
                fmha_config=fmha_config,
            )
            to.acquire()
            sp0.acquire()
            sp0.p_read()
            to.pv_mma(
                desc_v_base,
                FmhaStage.Head,
                0,
                False,
                fmha_config=fmha_config,
            )
            to.commit()
            kv_tile_end = sp0.init_kv_tile_idx()

            # LOOP: QK0(Ki), PV1(Vi-1), QK1(Ki), PV0(Vi).
            def mma_body(desc_v_base, kv_tile_end):
                kv_tile_end = sp0.next_kv_tile_idx()
                skv.wait()
                desc_k_base = skv.k_desc(
                    sK_array,
                    fmha_config=fmha_config,
                )
                sp0.qk_mma(
                    desc_q0_base,
                    desc_k_base,
                    FmhaStage.Loop,
                    0,
                    fmha_config=fmha_config,
                )
                sp0.commit()

                to.acquire()
                sp1.acquire()
                sp1.p_read()
                to.pv_mma(
                    desc_v_base,
                    FmhaStage.Loop,
                    0,
                    False,
                    fmha_config=fmha_config,
                )
                to.commit()
                skv.release()

                sp1.qk_mma(
                    desc_q1_base,
                    desc_k_base,
                    FmhaStage.Loop,
                    1,
                    fmha_config=fmha_config,
                )
                sp1.commit()
                skv.release()

                skv.wait()
                desc_v_base = skv.v_desc(
                    sK_array,
                    fmha_config=fmha_config,
                )
                to.acquire()
                sp0.acquire()
                sp0.p_read()
                to.pv_mma(
                    desc_v_base,
                    FmhaStage.Loop,
                    1,
                    False,
                    fmha_config=fmha_config,
                )
                to.commit()
                return desc_v_base, kv_tile_end

            desc_v_base, kv_tile_end = ts.domain_loop(
                loop_start,
                loop_end,
                loop_step,
                mma_body,
                desc_v_base,
                kv_tile_end,
            )

            # TAIL: close Q/SP0, then consume the last P1/V pair.
            sq.release()
            sq.release()
            sp0.commit()
            to.acquire()
            sp1.acquire()
            sp1.p_read()
            to.pv_mma(
                desc_v_base,
                FmhaStage.Tail,
                0,
                kv_tile_end,
                fmha_config=fmha_config,
            )
            to.commit()
            skv.release()
            sp1.commit()

    @ts.schedule
    def persistent_mma_schedule(stage_info, sq, skv, sp0, sp1, to, wq):
        mma_schedule_body(sq, skv, sp0, sp1, to, wq)

    @ts.schedule
    def direct_mma_schedule(stage_info, sq, skv, sp0, sp1, to):
        mma_schedule_body(sq, skv, sp0, sp1, to, None)

    schedule = (
        persistent_mma_schedule(
            smem_q,
            smem_kv,
            tmem_sp0,
            tmem_sp1,
            tmem_o,
            work_queue,
        )
        if work_queue is not None
        else direct_mma_schedule(
            smem_q,
            smem_kv,
            tmem_sp0,
            tmem_sp1,
            tmem_o,
        )
    )

    return task_class(
        src_resources=_src_resources(
            smem_q,
            smem_kv,
            tmem_sp0,
            tmem_sp1,
            work_queue=work_queue,
        ),
        dst_resources=[tmem_sp0, tmem_sp1, tmem_o],
        name="MmaTask",
        warp_idx=fmha_config.mma_warp_id,
        num_warps=1,
        schedule=schedule,
        num_registers=fmha_config.num_regs_other,
        **task_kwargs,
    )


def create_softmax_task(
    index,
    tmem_sp,
    tmem_vec,
    s0s1_seq,
    work_queue,
    *,
    fmha_config: FmhaConfig,
    tmem_vec_smem_offset,
    task_class=ts.Task,
    **task_kwargs,
):
    """Build one four-warp online-softmax task."""

    loop_start, loop_end, loop_step = _captured_loop_bounds(
        task_class, task_kwargs
    )
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)

    def softmax_schedule_body(sp, vec, seq, wq):
        sStats_array = vec.init_store_state(
            tmem_vec_smem_offset,
            fmha_config=fmha_config,
        )
        scale_softmax_log2 = sp.load_scale_softmax_log2()
        with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
            row_max, row_sum = sp.init_softmax_work_tile_state()
            kv_tile_end = sp.init_kv_tile_idx()
            q_offset = sp.cache_q_offset(fmha_config=fmha_config)
            if fmha_config.has_varlen and not fmha_config.is_causal:
                seqlen_k = sp.cache_seqlen_k()
            vec.acquire()

            def softmax_body(row_max, row_sum, kv_tile_end):
                kv_tile_idx = sp.current_kv_tile_idx()
                kv_tile_end = sp.next_kv_tile_idx()
                sp.wait()
                # Pass scores explicitly from row-max computation to exp2_p.
                if fmha_config.window_size_left > 0:
                    old_row_max, row_max, s_data = sp.left_masked_row_max(
                        row_max,
                        q_offset,
                        index,
                        fmha_config=fmha_config,
                    )
                elif (
                    fmha_config.is_causal
                    and fmha_config.head_paired
                    and fmha_config.has_q_offset
                ):
                    old_row_max, row_max, s_data = sp.right_masked_row_max(
                        row_max,
                        q_offset,
                        kv_tile_idx,
                        index,
                        fmha_config=fmha_config,
                    )
                elif (
                    fmha_config.is_causal
                    and not fmha_config.head_paired
                    and fmha_config.has_q_offset
                ):
                    old_row_max, row_max, s_data = (
                        sp.query_paired_masked_row_max(
                            row_max,
                            q_offset,
                            kv_tile_idx,
                            index,
                            fmha_config=fmha_config,
                        )
                    )
                elif fmha_config.has_varlen and not fmha_config.is_causal:
                    old_row_max, row_max, s_data = (
                        sp.packed_dense_k_masked_row_max(
                            row_max,
                            seqlen_k,
                            index,
                            fmha_config=fmha_config,
                        )
                    )
                else:
                    old_row_max, row_max, s_data = sp.compute_row_max(
                        row_max,
                        index,
                        fmha_config=fmha_config,
                    )
                vec.store_vec(
                    sStats_array,
                    old_row_max,
                    row_max,
                    row_sum,
                    False,
                    fmha_config=fmha_config,
                )
                vec.commit()

                if index == 0:
                    seq.acquire()
                else:
                    seq.wait()
                p_chunk = sp.exp2_p(
                    row_max,
                    scale_softmax_log2,
                    s_data,
                    index,
                    fmha_config=fmha_config,
                )
                if index == 0:
                    seq.commit()
                else:
                    seq.release()
                sp.release()
                row_sum = sp.softmax_aux_reduce(
                    old_row_max,
                    row_max,
                    row_sum,
                    p_chunk,
                    scale_softmax_log2,
                )
                vec.acquire()
                return row_max, row_sum, kv_tile_end

            row_max, row_sum, kv_tile_end = ts.domain_loop(
                loop_start,
                loop_end,
                loop_step,
                softmax_body,
                row_max,
                row_sum,
                kv_tile_end,
            )

            if fmha_config.is_causal:
                sp.wait()
                if fmha_config.head_paired:
                    old_row_max, row_max, s_data = sp.right_masked_row_max(
                        row_max,
                        q_offset,
                        kv_tile_end,
                        index,
                        fmha_config=fmha_config,
                    )
                else:
                    old_row_max, row_max, s_data = (
                        sp.query_paired_masked_row_max(
                            row_max,
                            q_offset,
                            kv_tile_end,
                            index,
                            fmha_config=fmha_config,
                        )
                    )
                vec.store_vec(
                    sStats_array,
                    old_row_max,
                    row_max,
                    row_sum,
                    False,
                    fmha_config=fmha_config,
                )
                vec.commit()
                if index == 0:
                    seq.acquire()
                else:
                    seq.wait()
                p_chunk = sp.exp2_p(
                    row_max,
                    scale_softmax_log2,
                    s_data,
                    index,
                    fmha_config=fmha_config,
                )
                if index == 0:
                    seq.commit()
                else:
                    seq.release()
                sp.release()
                row_sum = sp.softmax_aux_reduce(
                    old_row_max,
                    row_max,
                    row_sum,
                    p_chunk,
                    scale_softmax_log2,
                )
                if fmha_config.skip_causal_invalid_peer0 and index == 0:
                    sp.wait()
                    old_row_max, row_max = sp.invalid_row_max(row_max)
                    vec.acquire()
                    vec.store_vec(
                        sStats_array,
                        old_row_max,
                        row_max,
                        row_sum,
                        False,
                        fmha_config=fmha_config,
                    )
                    vec.commit()
                    seq.acquire()
                    sp.invalid_exp2_p(row_max)
                    seq.commit()
                    sp.release()
                sp.wait()
                sp.release()
                vec.acquire()
            else:
                # Consume the MMA sentinel; the reserved vec slot is final.
                sp.wait()
                sp.release()
            old_row_max = sp.softmax_aux_identity(row_max)
            vec.store_vec(
                sStats_array,
                old_row_max,
                row_max,
                row_sum,
                True,
                fmha_config=fmha_config,
            )
            vec.commit()

    @ts.schedule
    def persistent_softmax_schedule(stage_info, sp, vec, seq, wq):
        softmax_schedule_body(sp, vec, seq, wq)

    @ts.schedule
    def direct_softmax_schedule(stage_info, sp, vec, seq):
        softmax_schedule_body(sp, vec, seq, None)

    schedule = (
        persistent_softmax_schedule(tmem_sp, tmem_vec, s0s1_seq, work_queue)
        if work_queue is not None
        else direct_softmax_schedule(tmem_sp, tmem_vec, s0s1_seq)
    )

    return task_class(
        name=f"Softmax{index}Task",
        warp_idx=(
            fmha_config.softmax0_warp_ids[0]
            if index == 0
            else fmha_config.softmax1_warp_ids[0]
        ),
        num_warps=len(
            fmha_config.softmax0_warp_ids
            if index == 0
            else fmha_config.softmax1_warp_ids
        ),
        schedule=schedule,
        num_registers=fmha_config.num_regs_softmax,
        **task_kwargs,
    )


def create_correction_task(
    tmem_vec0,
    tmem_vec1,
    tmem_o,
    smem_o0,
    smem_o1,
    work_queue,
    *,
    fmha_config: FmhaConfig,
    tmem_vec0_smem_offset,
    tmem_vec1_smem_offset,
    o0_smem_offset,
    o1_smem_offset,
    task_class=ts.Task,
    **task_kwargs,
):
    """Build alternating O0/O1 correction and final output staging."""

    loop_start, loop_end, loop_step = _captured_loop_bounds(
        task_class, task_kwargs
    )
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)

    def correction_schedule_body(v0, v1, to, so0, so1, wq):
        sStats0_array = v0.init_read_state(
            tmem_vec0_smem_offset,
            fmha_config=fmha_config,
        )
        sStats1_array = v1.init_read_state(
            tmem_vec1_smem_offset,
            fmha_config=fmha_config,
        )
        sO0_array = so0.init_store_state(
            o0_smem_offset,
            fmha_config=fmha_config,
        )
        sO1_array = so1.init_store_state(
            o1_smem_offset,
            fmha_config=fmha_config,
        )
        scale_softmax_log2_v0 = v0.load_scale_softmax_log2()
        scale_softmax_log2_v1 = v1.load_scale_softmax_log2()
        output_scale0 = v0.load_output_scale()
        output_scale1 = v1.load_output_scale()
        with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
            # Discard the first stats0 record and retain stats1 for the first
            # cross-release, matching the source's alternating cadence.
            v0.wait()
            v0.release()
            v1.wait()

            def correction_body():
                v0.wait()
                vec_old_max0, vec_new_max0, _, vec_scale0 = v0.read_vec(
                    sStats0_array,
                    scale_softmax_log2_v0,
                    False,
                    fmha_config=fmha_config,
                )
                to.wait()
                to.correct(
                    vec_old_max0,
                    vec_new_max0,
                    vec_scale0,
                    0,
                    fmha_config=fmha_config,
                )
                v1.release()
                to.release()

                v1.wait()
                vec_old_max1, vec_new_max1, _, vec_scale1 = v1.read_vec(
                    sStats1_array,
                    scale_softmax_log2_v1,
                    False,
                    fmha_config=fmha_config,
                )
                to.wait()
                to.correct(
                    vec_old_max1,
                    vec_new_max1,
                    vec_scale1,
                    1,
                    fmha_config=fmha_config,
                )
                v0.release()
                to.release()

            ts.domain_loop(loop_start, loop_end, loop_step, correction_body)

            v1.release()
            v0.wait()
            _, _, vec_row_sum0, _ = v0.read_vec(
                sStats0_array,
                scale_softmax_log2_v0,
                True,
                fmha_config=fmha_config,
            )
            v0.release()
            to.wait()
            so0.acquire()
            so0.store_o(
                sO0_array,
                vec_row_sum0,
                output_scale0,
                0,
                fmha_config=fmha_config,
            )
            so0.commit()
            to.release()

            v1.wait()
            _, _, vec_row_sum1, _ = v1.read_vec(
                sStats1_array,
                scale_softmax_log2_v1,
                True,
                fmha_config=fmha_config,
            )
            v1.release()
            to.wait()
            so1.acquire()
            so1.store_o(
                sO1_array,
                vec_row_sum1,
                output_scale1,
                1,
                fmha_config=fmha_config,
            )
            so1.commit()
            to.release()

    @ts.schedule
    def persistent_correction_schedule(stage_info, v0, v1, to, so0, so1, wq):
        correction_schedule_body(v0, v1, to, so0, so1, wq)

    @ts.schedule
    def direct_correction_schedule(stage_info, v0, v1, to, so0, so1):
        correction_schedule_body(v0, v1, to, so0, so1, None)

    schedule = (
        persistent_correction_schedule(
            tmem_vec0,
            tmem_vec1,
            tmem_o,
            smem_o0,
            smem_o1,
            work_queue,
        )
        if work_queue is not None
        else direct_correction_schedule(
            tmem_vec0,
            tmem_vec1,
            tmem_o,
            smem_o0,
            smem_o1,
        )
    )

    return task_class(
        name="CorrectionTask",
        warp_idx=fmha_config.correction_warp_ids[0],
        num_warps=len(fmha_config.correction_warp_ids),
        schedule=schedule,
        num_registers=fmha_config.num_regs_correction,
        **task_kwargs,
    )


def create_epilogue_task(
    smem_o0,
    smem_o1,
    gmem_o0,
    gmem_o1,
    work_queue,
    *,
    fmha_config: FmhaConfig,
    o0_smem_offset,
    o1_smem_offset,
    task_class=ts.Task,
    **task_kwargs,
):
    """Build the one-warp paired TMA-store epilogue."""

    loop_start, loop_end, loop_step = _captured_loop_bounds(
        task_class, task_kwargs
    )
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)

    def epilogue_schedule_body(so0, so1, go0, go1, wq):
        sO0_array = so0.init_output_state(
            o0_smem_offset,
            fmha_config=fmha_config,
        )
        sO1_array = so1.init_output_state(
            o1_smem_offset,
            fmha_config=fmha_config,
        )
        with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
            with ts.domain_loop(loop_start, loop_end, loop_step):
                pass

            so0.wait()
            head_coord, batch_coord, seq_coord_q = so0.compute_output_coords(
                fmha_config=fmha_config
            )
            go0.tma_store(
                sO0_array,
                head_coord,
                batch_coord,
                seq_coord_q,
                0,
                fmha_config=fmha_config,
            )
            so0.release()

            so1.wait()
            head_coord, batch_coord, seq_coord_q = so1.compute_output_coords(
                fmha_config=fmha_config
            )
            go1.tma_store(
                sO1_array,
                head_coord,
                batch_coord,
                seq_coord_q,
                1,
                fmha_config=fmha_config,
            )
            so1.release()

    @ts.schedule
    def persistent_epilogue_schedule(stage_info, so0, so1, go0, go1, wq):
        epilogue_schedule_body(so0, so1, go0, go1, wq)

    @ts.schedule
    def direct_epilogue_schedule(stage_info, so0, so1, go0, go1):
        epilogue_schedule_body(so0, so1, go0, go1, None)

    schedule = (
        persistent_epilogue_schedule(
            smem_o0,
            smem_o1,
            gmem_o0,
            gmem_o1,
            work_queue,
        )
        if work_queue is not None
        else direct_epilogue_schedule(smem_o0, smem_o1, gmem_o0, gmem_o1)
    )

    return task_class(
        name="EpilogueTask",
        warp_idx=fmha_config.epilogue_warp_id,
        num_warps=1,
        schedule=schedule,
        num_registers=fmha_config.num_regs_other,
        **task_kwargs,
    )


def create_scheduler_task(work_queue, *, fmha_config: FmhaConfig):
    """Build the warp-15 CLC fetch producer."""

    @ts.schedule
    def scheduler_schedule(stage_info, wq):
        with ts.work_tile_loop(wq):
            with ts.domain_loop(0):
                pass
            wq.acquire()
            wq.fetch_work_tile()
            wq.commit()
            _persistent_tail(wq)

    return ts.Task(
        name="SchedulerTask",
        warp_idx=fmha_config.empty_warp_id,
        num_warps=1,
        schedule=scheduler_schedule(work_queue),
        num_registers=fmha_config.num_regs_other,
    )


def create_padding_task(
    work_queue,
    *,
    fmha_config: FmhaConfig,
    task_class=ts.Task,
    **task_kwargs,
):
    """Keep warp 15 in the final register group outside CLC scheduling."""

    loop_start, loop_end, loop_step = _captured_loop_bounds(
        task_class, task_kwargs
    )
    skip_work_tile_if = _packed_context_skip_predicate(work_queue)

    def padding_schedule_body(wq):
        with _work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
            with ts.domain_loop(loop_start, loop_end, loop_step):
                pass

    @ts.schedule
    def persistent_padding_schedule(stage_info, wq):
        padding_schedule_body(wq)

    @ts.schedule
    def direct_padding_schedule(stage_info):
        padding_schedule_body(None)

    schedule = (
        persistent_padding_schedule(work_queue)
        if work_queue is not None
        else direct_padding_schedule()
    )
    return task_class(
        name="PaddingTask",
        warp_idx=fmha_config.empty_warp_id,
        num_warps=1,
        schedule=schedule,
        num_registers=fmha_config.num_regs_other,
        **task_kwargs,
    )
