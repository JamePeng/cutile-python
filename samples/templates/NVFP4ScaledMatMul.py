# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import cuda.tile as ct
import torch
import sys
from cuda.tile._cext import get_compute_capability
from cuda.tile._bytecode.version import BytecodeVersion

from test.kernels.kernel_utils import swizzle_32_4_4, get_tileiras_version, \
                                      unpack_e2m1_bytes_to_float, block_quantize_f4e2m1fn_f8e4m3fn
from test.kernels.scaled_matmul import nvfp4_block_scaled_matmul_kernel


def cutile_nvfp4_matmul(A: torch.Tensor, A_scale: torch.Tensor, A_global: torch.Tensor,
                        B: torch.Tensor, B_scale: torch.Tensor, B_global: torch.Tensor,
                        scaling_block_size: int) -> torch.Tensor:

    """
    Performs NVFP4 matrix multiplication using a cuTile kernel.

    This wrapper function handles input validation, calculates the necessary grid dimensions,
    and launches the `nvfp4_block_scaled_matmul_kernel`.

    Args:
        A (torch.Tensor):         Packed input matrix with physical shape (M, K // 2).
                                  Must be on a CUDA device.
        A_scale (torch.Tensor):   Either 2D scale with shape (M, K // NVFP4_BLOCK_SIZE) or swizzled
                                  scale of (M // 32 // 4, K // NVFP4_BLOCK_SIZE // 4, 32, 16).
        A_global (torch.Tensor):  One-element float32 tensor containing A's tensor-wide scale.
        B (torch.Tensor):         Packed input matrix with physical shape (N, K // 2).
                                  Must be on a CUDA device and have its 2nd dimension match
                                  A's 2nd dimension.
        B_scale (torch.Tensor):   Either 2D scale with shape (N, K // NVFP4_BLOCK_SIZE) or swizzled
                                  scale of  (N // 32 // 4, K // NVFP4_BLOCK_SIZE // 4, 32, 16).
        B_global (torch.Tensor):  One-element float32 tensor containing B's tensor-wide scale.
        scaling_block_size (int): The scaling block size.

    Returns:
        torch.Tensor: The resulting matrix C (M x N) on the CUDA device.

    Raises:
        ValueError: If matrices are incompatible (K dimensions don't match),
                    or if they are not on a CUDA device.
    """
    # --- Input Validation ---
    if A.shape[1] != B.shape[1]:
        raise ValueError("Incompatible matrices")
    if not (A.device == A_scale.device == A_global.device ==
            B.device == B_scale.device == B_global.device):
        raise ValueError("Input tensors must be on the same device.")
    if not (A.is_cuda and A_scale.is_cuda and A_global.is_cuda and
            B.is_cuda and B_scale.is_cuda and B_global.is_cuda):
        raise ValueError("Input tensors must be on a CUDA device.")

    A = A.view(torch.uint8)
    B = B.view(torch.uint8)

    # Note: cuTile handles dtype compatibility within the kernel,
    # but inputs should generally match.
    tm, tn, tk = 256, 256, 256

    # --- Get Matrix Dimensions ---
    m, _ = A.shape
    n, _ = B.shape

    # --- Calculate Grid Dimensions for Kernel Launch (1D Grid) ---
    # The grid defines how many CUDA blocks (CTAs) will be launched.
    # Each block computes one (tm x tn) output tile of matrix C.
    # `ct.cdiv(total_dim, tile_dim)` ensures enough blocks are launched to cover
    # the entire matrix, even if dimensions are not perfect multiples of tile sizes.
    grid_x = ct.cdiv(m, tm)  # Number of blocks needed along the M dimension (rows of C)
    grid_y = ct.cdiv(n, tn)  # Number of blocks needed along the N dimension (columns of C)
    grid_size = grid_x * grid_y

    grid = (grid_size, 1, 1)

    # --- Create Output Tensor C ---
    # The output tensor `C` is initialized with the correct dimensions (M x N),
    # on the same device, and with a datatype of float32.
    C = torch.empty((m, n), device=A.device, dtype=torch.float32)

    # --- Launch the cuTile Kernel ---
    # The `nvfp4_block_scaled_matmul_kernel` is launched with the calculated grid dimensions.
    # `tm`, `tn`, and `tk` are passed as Constant integers to the kernel.
    kernel = nvfp4_block_scaled_matmul_kernel
    ct.launch(torch.cuda.current_stream(), grid, kernel, (
        A, A_scale, A_global, B, B_scale, B_global, C, tm, tn, tk, scaling_block_size))
    return C


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
    )
    args = parser.parse_args()

    if get_compute_capability()[0] < 10:
        print("Skipped test: NOT Running cuTile NVFP4 Matrix Multiplication Examples "
              "Blackwell or newer required.")
        sys.exit(0)

    if get_tileiras_version() < BytecodeVersion.V_13_4:
        print("Skipped test: NOT Running cuTile NVFP4 Matrix Multiplication Examples "
              "tileiras version 13.4 required.")
        sys.exit(0)

    # --- Running cuTile NVFP4 Matrix Multiplication Examples ---
    print("--- Running cuTile NVFP4 Matrix Multiplication Examples ---")

    # Define common matrix dimensions for the examples
    M_dim = 512
    N_dim = 512
    K_dim = 768

    scaling_block_size = 16

    print(f"\n--- Test Case: NVFP4 Matrix Multiplication with M = {M_dim}, N = {N_dim}, "
          f"K = {K_dim}, Scaling Block Size = {scaling_block_size} ---")

    A = torch.rand((M_dim, K_dim), device='cuda:0')
    B = torch.rand((N_dim, K_dim), device='cuda:0')

    uncompressed_ref = A @ B.T

    A, A_scale, A_global = block_quantize_f4e2m1fn_f8e4m3fn(A, scaling_block_size)
    B, B_scale, B_global = block_quantize_f4e2m1fn_f8e4m3fn(B, scaling_block_size)

    A_s_swizzled = swizzle_32_4_4(A_scale)
    B_s_swizzled = swizzle_32_4_4(B_scale)

    print(f"Input A packed shape: {A.shape}, dtype: {A.dtype}")
    print(f"Input B packed shape: {B.shape}, dtype: {B.dtype}")

    kernel_atol, kernel_rtol = 1e-4, 1e-3
    compression_atol, compression_rtol = 0.5, 0.05

    # Perform NVFP4 matrix multiplication using the cuTile wrapper function.
    C_cutile = cutile_nvfp4_matmul(A, A_scale, A_global, B, B_scale, B_global, scaling_block_size)
    torch.cuda.synchronize()
    C_cutile_swizzled = cutile_nvfp4_matmul(A, A_s_swizzled, A_global, B, B_s_swizzled, B_global,
                                            scaling_block_size)
    torch.cuda.synchronize()
    print(f"cuTile Output C shape: {C_cutile.shape}, dtype: {C_cutile.dtype}")

    if args.correctness_check:
        ref_A_scale = torch.repeat_interleave(A_scale, scaling_block_size, dim=1).to(torch.float32)
        ref_B_scale = torch.repeat_interleave(B_scale, scaling_block_size, dim=1).to(torch.float32)
        ref_A = unpack_e2m1_bytes_to_float(A.view(torch.uint8))
        ref_B = unpack_e2m1_bytes_to_float(B.view(torch.uint8))
        ref = (ref_A * ref_A_scale * A_global) @ (ref_B.T * ref_B_scale.T * B_global)

        torch.testing.assert_close(C_cutile, ref, atol=kernel_atol, rtol=kernel_rtol)
        torch.testing.assert_close(C_cutile_swizzled, ref, atol=kernel_atol, rtol=kernel_rtol)

        torch.testing.assert_close(C_cutile, uncompressed_ref,
                                   atol=compression_atol, rtol=compression_rtol)
        torch.testing.assert_close(C_cutile_swizzled, uncompressed_ref,
                                   atol=compression_atol, rtol=compression_rtol)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- cuTile NVFP4 matrix multiplication example completed. ---")
