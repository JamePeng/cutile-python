/*
 * SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "xla/ffi/api/c_api.h"

XLA_FFI_TypeId* cutile_call_state_type_id();
const XLA_FFI_TypeInfo* cutile_call_state_type_info();

XLA_FFI_Error* cutile_call_handler(XLA_FFI_CallFrame* cf);
