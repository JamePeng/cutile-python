// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "pynvvm.h"
#include <nvvm.h>

#define FOREACH_NVVM_FUNCTION_TO_LOAD(X) \
    X(nvvmGetErrorString) \
    X(nvvmIRVersion) \
    X(nvvmVersion) \
    X(nvvmCreateProgram) \
    X(nvvmDestroyProgram) \
    X(nvvmAddModuleToProgram) \
    X(nvvmCompileProgram) \
    X(nvvmGetProgramLog) \
    X(nvvmGetProgramLogSize) \
    X(nvvmGetCompiledResult) \
    X(nvvmGetCompiledResultSize)


#define DECLARE_NVVM_FUNCTION_PTR(name) \
    decltype(::name)* name = nullptr;

namespace { struct NVVM {
    PyPtr cdll;
    FOREACH_NVVM_FUNCTION_TO_LOAD(DECLARE_NVVM_FUNCTION_PTR);

    static PyTypeObject pytype;
}; }

namespace { struct NVVMProgram {
    PyPtr nvvm;
    nvvmProgram program = nullptr;

    ~NVVMProgram() {
        if (program)
            py_unwrap<NVVM>(nvvm.get()).nvvmDestroyProgram(&program);
    }

    static PyTypeObject pytype;
}; }


static void nvvm_result_string_append(const NVVM& nvvm, nvvmResult res, StringBuilder* builder) {
    builder->append(" (");
    const char* err_str = nvvm.nvvmGetErrorString(res);
    if (err_str) {
        builder->append_many(err_str, " = ", res);
    } else {
        builder->append_many("Unknown NVVM error code ", res);
    }
    builder->append(")");
}

template <typename... Args>
static ErrorRaised_t raise_nvvm_error(const NVVM& nvvm, nvvmResult res, Args&&... message) {
    StringBuilder builder;
    builder.append_many(std::forward<Args>(message)...);
    nvvm_result_string_append(nvvm, res, &builder);
    return builder.raise(PyExc_RuntimeError);
}

static PyObject* NVVM_ir_version(PyObject* self, PyObject* dummy) {
    NVVM& nvvm = py_unwrap<NVVM>(self);
    int major_ir, minor_ir, major_dbg, minor_dbg;
    nvvmResult res = nvvm.nvvmIRVersion(&major_ir, &minor_ir, &major_dbg, &minor_dbg);
    if (res != NVVM_SUCCESS) {
        raise_nvvm_error(nvvm, res, "nvvmIRVersion() failed");
        return nullptr;
    }
    return Py_BuildValue("(iiii)", major_ir, minor_ir, major_dbg, minor_dbg);
}

static PyObject* NVVM_version(PyObject* self, PyObject* dummy) {
    NVVM& nvvm = py_unwrap<NVVM>(self);
    int major, minor;
    nvvmResult res = nvvm.nvvmVersion(&major, &minor);
    if (res != NVVM_SUCCESS) {
        raise_nvvm_error(nvvm, res, "nvvmVersion() failed");
        return nullptr;
    }
    return Py_BuildValue("(ii)", major, minor);
}

static PyObject* NVVM_create_program(PyObject* self, PyObject* dummy) {
    NVVM& nvvm = py_unwrap<NVVM>(self);

    nvvmProgram prog;
    nvvmResult res = nvvm.nvvmCreateProgram(&prog);
    if (res != NVVM_SUCCESS) {
        raise_nvvm_error(nvvm, res, "Failed to create an NVVM program");
        return nullptr;
    }

    PyObject* ret = NVVMProgram::pytype.tp_new(&NVVMProgram::pytype, nullptr, nullptr);
    if (!ret) return {};

    NVVMProgram& prog_obj = py_unwrap<NVVMProgram>(ret);
    prog_obj.nvvm = newref(self);
    prog_obj.program = prog;
    return ret;
}

static PyMethodDef NVVM_methods[] = {
    {"ir_version", NVVM_ir_version, METH_NOARGS, nullptr},
    {"version", NVVM_version, METH_NOARGS, nullptr},
    {"create_program", NVVM_create_program, METH_NOARGS, nullptr},
    {}
};

static void* get_symbol(PyObject* cdll, const char* name, PyObject* ctypes) {
    PyPtr func = getattr(cdll, name);
    if (!func) return nullptr;

    PyPtr c_void_p = getattr(ctypes, "c_void_p");
    if (!c_void_p) return nullptr;

    PyPtr func_void_p = steal(PyObject_CallMethod(
                ctypes, "cast", "(OO)", func.get(), c_void_p.get()));
    if (!func_void_p) return nullptr;

    PyPtr ptr_value = getattr(func_void_p, "value");
    if (!ptr_value) return nullptr;

    void* ptr = PyLong_AsVoidPtr(ptr_value.get());
    if (PyErr_Occurred()) return nullptr;

    if (!ptr)
        raise(PyExc_ValueError, "Unexpected null pointer");
    return ptr;
}

#define GET_NVVM_SYMBOL(name) \
    if (!(nvvm.name = reinterpret_cast<decltype(nvvm.name)>( \
                get_symbol(cdll.get(), #name, ctypes.get())))) \
        return -1;

static int NVVM_init(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyObject* dll_path;
    const char* keywords[] = {"", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O", const_cast<char**>(keywords), &dll_path))
        return -1;

    PyPtr ctypes = steal(PyImport_ImportModule("ctypes"));
    if (!ctypes) return -1;

    PyPtr cdll = steal(PyObject_CallMethod(ctypes.get(), "CDLL", "(O)", dll_path));
    if (!cdll) return -1;

    NVVM& nvvm = py_unwrap<NVVM>(self);
    nvvm.cdll = cdll;
    FOREACH_NVVM_FUNCTION_TO_LOAD(GET_NVVM_SYMBOL)
    return 0;
}


PyTypeObject NVVM::pytype = {
    .tp_name = "cuda.tile._cext.NVVM",
    .tp_basicsize = sizeof(PythonWrapper<NVVM>),
    .tp_dealloc = pywrapper_dealloc<NVVM>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = NVVM_methods,
    .tp_init = NVVM_init,
    .tp_new = pywrapper_new<NVVM>,
};

static PyObject* NVVMProgram_add_module(PyObject* self, PyObject* args) {
    PyObject* contents = nullptr;
    const char* name = nullptr;
    if (!PyArg_ParseTuple(args, "Os", &contents, &name))
        return nullptr;

    const char* buffer;
    size_t size;
    if (PyBytes_Check(contents)) {
        buffer = PyBytes_AS_STRING(contents);
        size = PyBytes_GET_SIZE(contents);
    } else if (PyByteArray_Check(contents)) {
        buffer = PyByteArray_AS_STRING(contents);
        size = PyByteArray_GET_SIZE(contents);
    } else {
        raise(PyExc_TypeError, "Expected a bytes or a bytearray object as the first argument");
        return nullptr;
    }

    NVVMProgram& prog = py_unwrap<NVVMProgram>(self);
    NVVM& nvvm = py_unwrap<NVVM>(prog.nvvm.get());

    nvvmResult res = nvvm.nvvmAddModuleToProgram(prog.program, buffer, size, name);
    if (res != NVVM_SUCCESS) {
        raise_nvvm_error(nvvm, res, "nvvmAddModuleToProgram() failed");
        return nullptr;
    }

    return Py_NewRef(Py_None);
}

static std::optional<Vec<char>> try_get_program_log(const NVVM& nvvm, nvvmProgram prog) {
    size_t size = -1;
    nvvmResult res = nvvm.nvvmGetProgramLogSize(prog, &size);
    if (res != NVVM_SUCCESS)
        return std::nullopt;

    Vec<char> buffer(size);
    res = nvvm.nvvmGetProgramLog(prog, buffer.data());
    if (res != NVVM_SUCCESS)
        return std::nullopt;

    return buffer;
}

static PyObject* NVVMProgram_compile(PyObject* self, PyObject* options) {
    if (!PySequence_Check(options)) {
        raise(PyExc_TypeError, "Expected a sequence of compiler options");
        return nullptr;
    }

    // Unpack options
    Py_ssize_t num_options = PySequence_Length(options);
    Vec<const char*> unpacked_options;
    unpacked_options.reserve(num_options);

    for (Py_ssize_t i = 0; i < num_options; ++i) {
        PyPtr py_opt = steal(PySequence_GetItem(options, i));
        if (!py_opt) return nullptr;

        if (!PyUnicode_Check(py_opt.get())) {
            raise(PyExc_TypeError, "Compiler options must be strings");
            return nullptr;
        }

        const char* option = PyUnicode_AsUTF8(py_opt.get());
        if (!option) return nullptr;

        unpacked_options.push_back(option);
    }

    // Compile the program
    NVVMProgram& prog = py_unwrap<NVVMProgram>(self);
    NVVM& nvvm = py_unwrap<NVVM>(prog.nvvm.get());
    nvvmResult res = nvvm.nvvmCompileProgram(
            prog.program, unpacked_options.size(), unpacked_options.data());
    if (res != NVVM_SUCCESS) {
        StringBuilder builder;
        builder.append("nvvmCompileProgram() failed");
        nvvm_result_string_append(nvvm, res, &builder);

        std::optional<Vec<char>> log = try_get_program_log(nvvm, prog.program);
        if (log.has_value()) {
            if (log->empty() || (*log)[0] == '\0')
                builder.append(". Compilation log is empty.");
            else
                builder.append_many(". Compilation log:\n", log->data());
        } else {
            builder.append(". Failed to obtain the compilation log.");
        }
        builder.raise(PyExc_RuntimeError);
        return nullptr;
    }

    // Get the result
    size_t result_size = -1;
    res = nvvm.nvvmGetCompiledResultSize(prog.program, &result_size);
    if (res != NVVM_SUCCESS) {
        raise_nvvm_error(nvvm, res, "nvvmGetCompiledResultSize() failed");
        return nullptr;
    }

    if (result_size == 0) {
        raise(PyExc_RuntimeError, "NVVM compilation returned an empty result");
        return nullptr;
    }

    PyPtr ret = steal(PyByteArray_FromStringAndSize("", 0));
    if (!ret) return nullptr;

    if (PyByteArray_Resize(ret.get(), result_size) < 0)
        return nullptr;

    char* data = PyByteArray_AS_STRING(ret.get());
    res = nvvm.nvvmGetCompiledResult(prog.program, data);
    if (res != NVVM_SUCCESS) {
        raise_nvvm_error(nvvm, res, "nvvmGetCompiledResult() failed");
        return nullptr;
    }

    if (data[result_size - 1] != '\0') {
        raise(PyExc_RuntimeError, "nvvmGetCompiledResult() did not return a NUL-terminated string");
        return nullptr;
    }

    if (PyByteArray_Resize(ret.get(), result_size - 1) < 0)
        return nullptr;

    return ret.release();
}

static PyMethodDef NVVMProgram_methods[] = {
    {"add_module", NVVMProgram_add_module, METH_VARARGS, nullptr},
    {"compile", NVVMProgram_compile, METH_O, nullptr},
    {}
};

PyTypeObject NVVMProgram::pytype = {
    .tp_name = "cuda.tile._cext.NVVMProgram",
    .tp_basicsize = sizeof(PythonWrapper<NVVMProgram>),
    .tp_dealloc = pywrapper_dealloc<NVVMProgram>,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_methods = NVVMProgram_methods,
    .tp_new = pywrapper_new<NVVMProgram>,
};

Status pynvvm_init(PyObject* m) {
    if (PyType_Ready(&NVVM::pytype) < 0)
        return ErrorRaised;

    if (PyModule_AddObjectRef(m, "NVVM",
                reinterpret_cast<PyObject*>(&NVVM::pytype)) < 0)
        return ErrorRaised;

    if (PyType_Ready(&NVVMProgram::pytype) < 0)
        return ErrorRaised;

    if (PyModule_AddObjectRef(m, "NVVMProgram",
                reinterpret_cast<PyObject*>(&NVVMProgram::pytype)) < 0)
        return ErrorRaised;

    return OK;
}
