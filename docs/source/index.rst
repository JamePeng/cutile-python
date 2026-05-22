.. SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
..
.. SPDX-License-Identifier: Apache-2.0

cuTile Python
=============

cuTile is a parallel programming model for NVIDIA GPUs and a Python-based :abbr:`DSL (Domain-Specific Language)`.
It automatically leverages advanced hardware capabilities, such as tensor cores and tensor memory accelerators,
while providing portability across different NVIDIA GPU architectures.
cuTile enables the latest hardware features without requiring code changes.

cuTile |kernels| are GPU programs that are executed in parallel on a logical |grid| of |blocks|.
The :py:class:`@ct.kernel <cuda.tile.kernel>` decorator marks a Python function as a kernel's entry point.
Kernels cannot be called directly from the host code; the host must queue kernels for execution on GPU
using the :py:func:`ct.launch() <cuda.tile.launch>` function:

.. testcode::

    # begin-snippet
    import cuda.tile as ct
    import cupy

    TILE_SIZE = 16

    # cuTile kernel for adding two dense vectors. It runs in parallel on the GPU.
    @ct.kernel
    def vector_add_kernel(a, b, result):
        block_id = ct.bid(0)
        a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
        b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
        result_tile = a_tile + b_tile
        ct.store(result, index=(block_id,), tile=result_tile)

    # Host-side function that launches the above kernel.
    def vector_add(a: cupy.ndarray, b: cupy.ndarray, result: cupy.ndarray):
        assert a.shape == b.shape == result.shape
        grid = (ct.cdiv(a.shape[0], TILE_SIZE), 1, 1)
        ct.launch(cupy.cuda.get_current_stream(), grid, vector_add_kernel, (a, b, result))
    # end-snippet

    import numpy as np

    rng = cupy.random.default_rng()
    a = rng.random(128)
    b = rng.random(128)
    result = cupy.zeros_like(a)

    vector_add(a, b, result)

    a_np = cupy.asnumpy(a)
    b_np = cupy.asnumpy(b)
    result_np = cupy.asnumpy(result)

    expected = a_np + b_np
    np.testing.assert_array_almost_equal(result_np, expected)

|Kernels| move data between |arrays| and |tiles| using functions like
:py:func:`ct.load() <cuda.tile.load>` and :py:func:`ct.store() <cuda.tile.store>`.
Both arrays and tiles are tensor-like data structures: each has a specific shape
(i.e., the number of elements along each axis) and a |dtype| (i.e., the data type of elements).
However, there are important differences:

- |Arrays| are stored in the global memory. They are mutable and have physical, strided
  memory layouts. Within the kernel code, they support only a limited set of operations,
  mostly related to |loading and storing| data to/from tiles. Various Python objects,
  including PyTorch tensors and CuPy arrays, can be passed as arrays from the host code
  to the kernel via kernel arguments.

- |Tiles| are immutable values without defined storage that only exist in the kernel code.
  Tile dimensions must be compile-time constants that are powers of two.
  Tiles support a multitude of |operations|, including elementwise arithmetic,
  matrix multiplication, reduction, shape manipulation, etc.

Proceed to the :ref:`quickstart` page for installation instructions and a complete working example.


.. toctree::
   :maxdepth: 2
   :hidden:

   generated/release_notes
   quickstart
   execution
   data
   memory_model
   interoperability
   performance
   operations
   compilation
   debugging
   known_issues
