/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "py.h"

#include <cstdint>


struct NativeLaunchSite;

struct NativeLaunchConfig {
    void* stream;
    int64_t grid[3];
    int64_t block[3];
    int64_t cluster[3];
    int64_t preferred_cluster[3];
    int32_t has_cluster;
    int32_t has_preferred_cluster;
    int32_t cooperative;
    int32_t programmatic_dependent_launch;
};

Result<NativeLaunchSite*> native_launch_site_create(
        PyObject* dispatcher,
        PyObject* const* pyargs,
        Py_ssize_t num_pyargs);
void native_launch_site_destroy(NativeLaunchSite* site);
int32_t native_launch_site_launch(
        NativeLaunchSite* site,
        const NativeLaunchConfig& config,
        void** argument_addresses);

Status tile_kernel_init(PyObject* m);
