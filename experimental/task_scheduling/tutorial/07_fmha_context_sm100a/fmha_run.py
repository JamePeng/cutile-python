# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Run and verify the SM100 paired FMHA scheduler variants."""

import argparse
import math
import statistics

import cuda.lang as cl
import torch

try:
    from .fmha_kernel import compute_grid, get_kernel, get_pipeline, make_tma_view
    from .fmha_resources import (
        ATTENTION_SCALE_LOG2,
        HEAD_DIM,
        SUPPORTED_SCHEDULERS,
        FmhaConfig,
    )
except ImportError:
    from fmha_kernel import compute_grid, get_kernel, get_pipeline, make_tma_view
    from fmha_resources import (
        ATTENTION_SCALE_LOG2,
        HEAD_DIM,
        SUPPORTED_SCHEDULERS,
        FmhaConfig,
    )


DEFAULT_Q_SHAPE = (1, 512, 2, HEAD_DIM)
DEFAULT_K_SHAPE = (1, 512, 2, HEAD_DIM)


def _validate_tensors(tensors, cfg=FmhaConfig()):
    required = ("q", "k", "v", "out", "scale", "output_scale")
    missing = [name for name in required if name not in tensors]
    if missing:
        raise ValueError(f"missing tensors: {missing}")
    q, k, v, out = (tensors[name] for name in ("q", "k", "v", "out"))
    expected_rank = 3 if cfg.has_varlen else 4
    if any(tensor.ndim != expected_rank for tensor in (q, k, v, out)):
        storage = "packed THD rank-3" if cfg.has_varlen else "BSHD rank-4"
        raise ValueError(f"Q, K, V, and O must use contiguous {storage} storage")
    if q.shape != out.shape:
        raise ValueError("O must have the same shape as Q")
    if k.shape != v.shape:
        raise ValueError("K and V must have identical shapes")
    if cfg.has_varlen:
        if q.shape[2] != k.shape[2] or q.shape[1] != k.shape[1] * cfg.h_r:
            raise ValueError(
                "packed ragged attention requires matching D and Hq == Hkv * h_r"
            )
        for name in ("qo_indptr", "kv_indptr"):
            if name not in tensors:
                raise ValueError(f"packed ragged FMHA requires {name}")
            indptr = tensors[name]
            if (
                indptr.ndim != 1
                or indptr.numel() < 2
                or indptr.dtype != torch.int32
                or not indptr.is_cuda
                or not indptr.is_contiguous()
            ):
                raise ValueError(
                    f"{name} must be a contiguous CUDA int32 tensor of shape [B+1]"
                )
        if tensors["qo_indptr"].numel() != tensors["kv_indptr"].numel():
            raise ValueError("qo_indptr and kv_indptr must describe the same batch")
    elif (
        q.shape[0] != k.shape[0]
        or q.shape[3] != k.shape[3]
        or q.shape[2] != k.shape[2] * cfg.h_r
    ):
        raise ValueError(
            "fixed attention requires matching B/D and Hq == Hkv * h_r"
        )
    if q.shape[-1] != HEAD_DIM:
        raise ValueError(f"the first port requires D={HEAD_DIM}")
    if any(tensor.dtype != torch.float16 for tensor in (q, k, v, out)):
        raise ValueError("the first port requires FP16 Q, K, V, and O")
    if any(not tensor.is_cuda for tensor in (q, k, v, out)):
        raise ValueError("Q, K, V, and O must be CUDA tensors")
    if any(not tensor.is_contiguous() for tensor in (q, k, v, out)):
        raise ValueError("Q, K, V, and O must be contiguous")
    for name in ("scale", "output_scale"):
        scale = tensors[name]
        if (
            scale.shape != (1,)
            or scale.dtype != torch.float32
            or not scale.is_cuda
            or not scale.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be a contiguous CUDA FP32 tensor of shape [1]"
            )
    return q, k, v, out


def prepare_tensors(
    q_shape=DEFAULT_Q_SHAPE,
    k_shape=DEFAULT_K_SHAPE,
    *,
    cfg=FmhaConfig(),
    seed=1111,
):
    if cfg.has_varlen:
        raise ValueError("use prepare_ragged_tensors() for packed ragged storage")
    if len(q_shape) != 4 or len(k_shape) != 4:
        raise ValueError("Q and K shapes must be B,S,H,D")
    batch, seq_q, heads, depth = q_shape
    batch_k, seq_k, heads_k, depth_k = k_shape
    if batch != batch_k or depth != depth_k or heads != heads_k * cfg.h_r:
        raise ValueError(
            "fixed attention requires matching B/D and Hq == Hkv * h_r"
        )
    # Reuse the kernel's strict tile-aligned geometry diagnostics.
    get_pipeline(batch, seq_q, seq_k, heads, cfg)
    torch.manual_seed(seed)

    def make(shape):
        return (torch.randn(shape, device="cuda", dtype=torch.float32) * 0.2).to(
            torch.float16
        )

    return {
        "q": make(q_shape),
        "k": make(k_shape),
        "v": make(k_shape),
        "out": torch.empty(q_shape, device="cuda", dtype=torch.float16),
        "scale": torch.tensor(
            [ATTENTION_SCALE_LOG2],
            device="cuda",
            dtype=torch.float32,
        ),
        "output_scale": torch.ones(1, device="cuda", dtype=torch.float32),
    }


def _cumulative(lengths):
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    return offsets


def prepare_ragged_tensors(
    q_lengths,
    k_lengths,
    *,
    heads=2,
    kv_heads=None,
    cfg=FmhaConfig(has_varlen=True),
    seed=1111,
):
    """Allocate packed THD tensors and live indptrs."""
    if not cfg.has_varlen:
        raise ValueError("prepare_ragged_tensors requires has_varlen=True")
    q_lengths = tuple(q_lengths)
    k_lengths = tuple(k_lengths)
    if not q_lengths or len(q_lengths) != len(k_lengths):
        raise ValueError("Q and K lengths must describe the same non-empty batch")
    if any(
        type(length) is not int or length <= 0 for length in (*q_lengths, *k_lengths)
    ):
        raise ValueError("all packed ragged sequence lengths must be positive integers")
    if cfg.is_causal and any(q_len > k_len for q_len, k_len in zip(q_lengths, k_lengths)):
        raise ValueError(
            "bottom-right causal attention requires every Q length <= K length"
        )
    if (
        cfg.is_causal
        and not cfg.has_q_offset
        and any(q_len != k_len for q_len, k_len in zip(q_lengths, k_lengths))
    ):
        raise ValueError(
            "unequal packed causal Q/K lengths require has_q_offset=True"
        )
    if type(heads) is not int or heads <= 0:
        raise ValueError("heads must be a positive integer")
    if heads % cfg.h_r:
        raise ValueError(f"heads ({heads}) must be divisible by h_r ({cfg.h_r})")
    expected_kv_heads = heads // cfg.h_r
    if kv_heads is None:
        kv_heads = expected_kv_heads
    if type(kv_heads) is not int or kv_heads != expected_kv_heads:
        raise ValueError(
            f"kv_heads must equal heads / h_r ({expected_kv_heads}), got {kv_heads}"
        )

    batch = len(q_lengths)
    max_seq_len_q = max(q_lengths)
    max_seq_len_k = max(k_lengths)
    get_pipeline(batch, max_seq_len_q, max_seq_len_k, heads, cfg)
    torch.manual_seed(seed)

    def make(total, num_heads):
        return (
            torch.randn(
                (total, num_heads, HEAD_DIM),
                device="cuda",
                dtype=torch.float32,
            )
            * 0.2
        ).to(torch.float16)

    q_offsets = _cumulative(q_lengths)
    k_offsets = _cumulative(k_lengths)
    return {
        "q": make(q_offsets[-1], heads),
        "k": make(k_offsets[-1], kv_heads),
        "v": make(k_offsets[-1], kv_heads),
        "out": torch.empty(
            (q_offsets[-1], heads, HEAD_DIM),
            device="cuda",
            dtype=torch.float16,
        ),
        "scale": torch.tensor(
            [ATTENTION_SCALE_LOG2],
            device="cuda",
            dtype=torch.float32,
        ),
        "output_scale": torch.ones(1, device="cuda", dtype=torch.float32),
        "qo_indptr": torch.tensor(q_offsets, device="cuda", dtype=torch.int32),
        "kv_indptr": torch.tensor(k_offsets, device="cuda", dtype=torch.int32),
        "_max_seq_len_q": max_seq_len_q,
        "_max_seq_len_k": max_seq_len_k,
    }


def run(
    tensors,
    cfg=FmhaConfig(),
    *,
    stream=None,
    verbose=False,
    max_seq_len_q=None,
    max_seq_len_k=None,
):
    q, k, v, out = _validate_tensors(tensors, cfg)
    if cfg.has_varlen:
        batch = tensors["qo_indptr"].numel() - 1
        heads = q.shape[1]
        if max_seq_len_q is None:
            max_seq_len_q = tensors.get("_max_seq_len_q")
        if max_seq_len_k is None:
            max_seq_len_k = tensors.get("_max_seq_len_k")
        if max_seq_len_q is None:
            max_seq_len_q = int(torch.diff(tensors["qo_indptr"]).max().item())
        if max_seq_len_k is None:
            max_seq_len_k = int(torch.diff(tensors["kv_indptr"]).max().item())
        seq_q = max_seq_len_q
        seq_k = max_seq_len_k
    else:
        batch, seq_q, heads, _ = q.shape
        seq_k = k.shape[1]

    pipeline = get_pipeline(batch, seq_q, seq_k, heads, cfg, verbose=verbose)
    kernel = get_kernel(pipeline)
    if stream is None:
        stream = torch.cuda.current_stream()
    arguments = (
        make_tma_view(q),
        make_tma_view(k),
        make_tma_view(v),
        make_tma_view(out),
        tensors["scale"],
        tensors["output_scale"],
    )
    if cfg.has_varlen:
        arguments += (tensors["qo_indptr"], tensors["kv_indptr"])
    cl.launch(
        stream,
        compute_grid(pipeline),
        (pipeline.cfg.block_threads,),
        kernel,
        arguments,
        block_in_cluster_count=pipeline.cfg.cluster_shape,
        programmatic_dependent_launch=False,
    )
    return out


def torch_reference(tensors, cfg=FmhaConfig()):
    q, k, v, _ = _validate_tensors(tensors, cfg)
    softmax_scale = tensors["scale"].item() / math.log2(math.e)
    output_scale = tensors["output_scale"].item()
    if cfg.has_varlen:
        q_offsets = tensors["qo_indptr"].cpu().tolist()
        k_offsets = tensors["kv_indptr"].cpu().tolist()
        expected = torch.empty_like(q, dtype=torch.float32)
        for batch in range(len(q_offsets) - 1):
            q_begin, q_end = q_offsets[batch:batch + 2]
            k_begin, k_end = k_offsets[batch:batch + 2]
            qh = q[q_begin:q_end].float().permute(1, 0, 2)
            kh = k[k_begin:k_end].float().permute(1, 0, 2)
            vh = v[k_begin:k_end].float().permute(1, 0, 2)
            if cfg.h_r > 1:
                kh = kh.repeat_interleave(cfg.h_r, dim=0)
                vh = vh.repeat_interleave(cfg.h_r, dim=0)
            scores = torch.matmul(qh, kh.transpose(-2, -1)) * softmax_scale
            if cfg.is_causal:
                q_idx = torch.arange(
                    q_end - q_begin, device=scores.device
                ).unsqueeze(1)
                k_idx = torch.arange(
                    k_end - k_begin, device=scores.device
                ).unsqueeze(0)
                q_offset = (k_end - k_begin) - (q_end - q_begin)
                mask = k_idx <= q_idx + q_offset
                if cfg.window_size_left > 0:
                    mask &= k_idx >= q_idx + q_offset - cfg.window_size_left
                scores = scores.masked_fill(~mask, -float("inf"))
            probability = torch.softmax(scores, dim=-1)
            expected[q_begin:q_end] = (
                torch.matmul(probability, vh).permute(1, 0, 2) * output_scale
            )
        return expected.to(torch.float16)
    qh = q.float().permute(0, 2, 1, 3)
    kh = k.float().permute(0, 2, 1, 3)
    vh = v.float().permute(0, 2, 1, 3)
    if cfg.h_r > 1:
        kh = kh.repeat_interleave(cfg.h_r, dim=1)
        vh = vh.repeat_interleave(cfg.h_r, dim=1)
    scores = torch.matmul(qh, kh.transpose(-2, -1)) * softmax_scale
    if cfg.is_causal:
        seq_q = q.shape[1]
        seq_k = k.shape[1]
        q_idx = torch.arange(seq_q, device=scores.device).unsqueeze(1)
        k_idx = torch.arange(seq_k, device=scores.device).unsqueeze(0)
        q_offset = seq_k - seq_q
        mask = k_idx <= q_idx + q_offset
        if cfg.window_size_left > 0:
            mask &= k_idx >= q_idx + q_offset - cfg.window_size_left
        scores = scores.masked_fill(~mask, -float("inf"))
    probability = torch.softmax(scores, dim=-1)
    return (torch.matmul(probability, vh).permute(0, 2, 1, 3) * output_scale).to(
        torch.float16
    )


def verify_output(tensors, *, cfg=FmhaConfig(), atol=3.0e-2, rtol=2.0e-2):
    expected = torch_reference(tensors, cfg)
    torch.testing.assert_close(tensors["out"], expected, atol=atol, rtol=rtol)
    error = (tensors["out"].float() - expected.float()).abs()
    return {
        "max_abs_error": error.max().item(),
        "mean_abs_error": error.mean().item(),
    }


def verify(
    q_shape=DEFAULT_Q_SHAPE,
    k_shape=DEFAULT_K_SHAPE,
    *,
    cfg=FmhaConfig(),
    seed=1111,
    verbose=False,
):
    tensors = prepare_tensors(q_shape, k_shape, cfg=cfg, seed=seed)
    run(tensors, cfg=cfg, verbose=verbose)
    torch.cuda.synchronize()
    metrics = verify_output(tensors, cfg=cfg)
    pairing = "head-paired GQA" if cfg.head_paired else "query-paired MHA"
    print(
        f"PASS: {cfg.scheduler}, {pairing}, D128 dual-instance "
        f"FMHA; q={q_shape}, k={k_shape}, max_abs={metrics['max_abs_error']:.6g}"
    )
    return metrics


def verify_ragged(
    q_lengths=(33, 257),
    k_lengths=(65, 257),
    *,
    heads=2,
    cfg=FmhaConfig(has_varlen=True),
    seed=1111,
    verbose=False,
):
    tensors = prepare_ragged_tensors(
        q_lengths,
        k_lengths,
        heads=heads,
        cfg=cfg,
        seed=seed,
    )
    run(tensors, cfg=cfg, verbose=verbose)
    torch.cuda.synchronize()
    metrics = verify_output(tensors, cfg=cfg)
    pairing = "head-paired GQA" if cfg.head_paired else "query-paired MHA"
    print(
        f"PASS: {cfg.scheduler}, packed-ragged {pairing} D128 "
        f"dual-instance FMHA; q={tuple(q_lengths)}, k={tuple(k_lengths)}, "
        f"max_abs={metrics['max_abs_error']:.6g}"
    )
    return metrics


def benchmark(
    q_shape=DEFAULT_Q_SHAPE,
    k_shape=DEFAULT_K_SHAPE,
    *,
    cfg=FmhaConfig(),
    warmups=20,
    iterations=50,
    repeats=5,
    seed=1111,
):
    if min(warmups, iterations, repeats) <= 0:
        raise ValueError("warmups, iterations, and repeats must be positive")
    tensors = prepare_tensors(q_shape, k_shape, cfg=cfg, seed=seed)
    for _ in range(warmups):
        run(tensors, cfg=cfg)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run(tensors, cfg=cfg)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    median_us = statistics.median(samples)
    print(f"median latency: {median_us:.3f} us; samples={samples}")
    return median_us, tuple(samples)


def benchmark_ragged(
    q_lengths,
    k_lengths,
    *,
    heads=2,
    cfg=FmhaConfig(has_varlen=True),
    warmups=20,
    iterations=50,
    repeats=5,
    seed=1111,
):
    if min(warmups, iterations, repeats) <= 0:
        raise ValueError("warmups, iterations, and repeats must be positive")
    tensors = prepare_ragged_tensors(
        q_lengths,
        k_lengths,
        heads=heads,
        cfg=cfg,
        seed=seed,
    )
    for _ in range(warmups):
        run(tensors, cfg=cfg)
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            run(tensors, cfg=cfg)
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / iterations)
    median_us = statistics.median(samples)
    print(f"median ragged latency: {median_us:.3f} us; samples={samples}")
    return median_us, tuple(samples)


def _parse_shape(value):
    try:
        shape = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "shape must be comma-separated integers"
        ) from error
    if len(shape) != 4:
        raise argparse.ArgumentTypeError("shape must contain B,S,H,D")
    return shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-shape", type=_parse_shape, default=DEFAULT_Q_SHAPE)
    parser.add_argument("--k-shape", type=_parse_shape, default=DEFAULT_K_SHAPE)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument(
        "--scheduler",
        choices=SUPPORTED_SCHEDULERS,
        default="clc_dynamic_persistent",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    arguments = parser.parse_args()
    cfg = FmhaConfig(scheduler=arguments.scheduler)
    verify(
        arguments.q_shape,
        arguments.k_shape,
        cfg=cfg,
        seed=arguments.seed,
        verbose=arguments.verbose,
    )
    if arguments.benchmark:
        benchmark(
            arguments.q_shape,
            arguments.k_shape,
            cfg=cfg,
            seed=arguments.seed,
        )


if __name__ == "__main__":
    main()
