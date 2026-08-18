# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Custom NVVM
if (CUSTOM_STATIC_LIBNVVM)
    add_library(nvvm SHARED nvvm_dummy.cpp)
    target_link_options(nvvm PRIVATE
        "-Wl,--whole-archive"
        ${CUSTOM_STATIC_LIBNVVM}
        "-Wl,--no-whole-archive")
endif()

# Custom ptxas
if (CUSTOM_PTXAS)
    set(dst "${CMAKE_CURRENT_BINARY_DIR}/ptxas${CMAKE_EXECUTABLE_SUFFIX}")
    add_custom_command(
        OUTPUT ${dst}
        COMMAND ${CMAKE_COMMAND} -E copy "${CUSTOM_PTXAS}" "${dst}"
        DEPENDS "${CUSTOM_PTXAS}"
    )
    add_custom_target(copy_custom_ptxas ALL DEPENDS ${dst})
endif()

