<!--
SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0
-->

Steps to build documentation

1. Install sphinx `pip install -r requirements.txt`
2. In the root of this repo run `pip install -e .`
3. Run `make html`
4. Open `build/html/index.html`
4. To run doctest, ``make doctest`
