<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- New `ct.pack_to_bytes()` operation that flattens a tile and reinterprets its
  raw bytes as a 1D uint8 tile.
- New `ct.unpack_from_bytes()` operation that reinterprets a 1D uint8 tile as a
  1D tile of the target dtype. Inverse of `ct.pack_to_bytes()`.
