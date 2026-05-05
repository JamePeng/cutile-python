// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "xla_ffi_py.h"
#include "xla_ffi.h"
#include "py.h"


namespace {

PyObject* py_get_call_handler(PyObject*, PyObject*) {
    return PyCapsule_New(reinterpret_cast<void*>(&cutile_call_handler),
                         nullptr, nullptr);
}

PyObject* py_get_call_type_id(PyObject*, PyObject*) {
    return PyCapsule_New(cutile_call_state_type_id(), nullptr, nullptr);
}

PyObject* py_get_call_type_info(PyObject*, PyObject*) {
    return PyCapsule_New(
        const_cast<XLA_FFI_TypeInfo*>(cutile_call_state_type_info()),
        nullptr, nullptr);
}

PyMethodDef functions[] = {
    {"xla_ffi_get_call_handler", &py_get_call_handler, METH_NOARGS, ""},
    {"xla_ffi_get_call_type_id", &py_get_call_type_id, METH_NOARGS, ""},
    {"xla_ffi_get_call_type_info", &py_get_call_type_info, METH_NOARGS, ""},
    {nullptr, nullptr, 0, nullptr},
};

}  // namespace


Status xla_ffi_init(PyObject* m) {
    if (PyModule_AddFunctions(m, functions) < 0)
        return ErrorRaised;
    return OK;
}
