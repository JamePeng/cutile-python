# SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set(XLA_REPOSITORY "https://github.com/openxla/xla" CACHE STRING "OpenXLA repo URL")
set(XLA_REVISION   "b6f37ab7767f428fd6f993de5e211643d47d4deb"
    CACHE STRING "XLA revision — (from jax v0.10)")

set(XLA_REVISION_SHA256 "6a25ffe712c27468fe642c8d07f4de5601092bf31f4d1edd483eeef96b39e2d9"
    CACHE STRING "Expected SHA256 of the pinned XLA archive")


macro(fetch_xla_headers)
  if(XLA_PATH)
    message(STATUS "Using local xla headers, include path: ${XLA_PATH}")
    set(xla_INCLUDE_DIR "${XLA_PATH}")
  else()
    include(FetchContent)
    message(STATUS "Fetching xla headers: ${XLA_REVISION}")
    FetchContent_Declare(
      xla
      URL           "${XLA_REPOSITORY}/archive/${XLA_REVISION}.tar.gz"
      URL_HASH      "SHA256=${XLA_REVISION_SHA256}"
      SOURCE_SUBDIR does-not-exist           # don't add_subdirectory
      DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(xla)
    set(xla_INCLUDE_DIR "${xla_SOURCE_DIR}")
  endif()
endmacro()
