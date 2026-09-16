<!--- SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0 -->

- Fixed a compilation failure when `range()` is given an `int64` bound or step.
  The bounds, step and induction variable are now promoted to a common integer type.
