// SPDX-FileCopyrightText: Copyright (c) <2026> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0

#include "bitstream.h"

#include "vec.h"

namespace { struct BitstreamWriter {
    Vec<uint32_t> words;
    uint64_t buf = 0;
    unsigned buf_size = 0;  // Number of bits in `buf`, always <= 63

    static PyTypeObject pytype;
}; }

static void append_fixed(BitstreamWriter* w, uint64_t bits, unsigned width) {
    unsigned buf_size = w->buf_size;
    uint64_t buf = w->buf | (bits << buf_size);
    buf_size += width;
    if (buf_size >= 64) {
        w->words.push_back(static_cast<uint32_t>(buf));
        w->words.push_back(static_cast<uint32_t>(buf >> 32));
        buf_size -= 64;
        buf = bits >> (width - buf_size);
    }
    w->buf = buf;
    w->buf_size = buf_size;
}

static void append_vbr(BitstreamWriter* w, uint64_t bits, unsigned width) {
    while (true) {
        uint64_t next = bits >> (width - 1);
        if (!next) break;
        uint64_t high_bit = (uint64_t(1) << (width - 1));
        append_fixed(w, (bits & (high_bit - 1)) | high_bit, width);
        bits = next;
    }
    append_fixed(w, bits, width);
}

static PyObject* py_append_fixed(PyObject* self, PyObject* py_bits, unsigned width) {
    BitstreamWriter& w = py_unwrap<BitstreamWriter>(self);
    uint64_t bits = pylong_as<uint64_t>(py_bits);
    if (PyErr_Occurred()) return nullptr;
    if (bits >> width) {
        raise(PyExc_OverflowError, "Value ", bits, " is too big for a ",
              width, "-bit unsigned integer");
        return nullptr;
    }
    append_fixed(&w, bits, width);
    return Py_NewRef(Py_None);
}

static PyObject* py_append_vbr(PyObject* self, PyObject* py_bits, unsigned width) {
    BitstreamWriter& w = py_unwrap<BitstreamWriter>(self);
    uint64_t bits = pylong_as<uint64_t>(py_bits);
    if (PyErr_Occurred()) return nullptr;
    append_vbr(&w, bits, width);
    return Py_NewRef(Py_None);
}

static PyObject* BitstreamWriter_fixed1(PyObject* self, PyObject* arg) {
    return py_append_fixed(self, arg, 1);
}

static PyObject* BitstreamWriter_fixed2(PyObject* self, PyObject* arg) {
    return py_append_fixed(self, arg, 2);
}

static PyObject* BitstreamWriter_fixed3(PyObject* self, PyObject* arg) {
    return py_append_fixed(self, arg, 3);
}

static PyObject* BitstreamWriter_fixed4(PyObject* self, PyObject* arg) {
    return py_append_fixed(self, arg, 4);
}

static PyObject* BitstreamWriter_fixed8(PyObject* self, PyObject* arg) {
    return py_append_fixed(self, arg, 8);
}

static PyObject* BitstreamWriter_fixed32(PyObject* self, PyObject* arg) {
    return py_append_fixed(self, arg, 32);
}

static PyObject* BitstreamWriter_vbr4(PyObject* self, PyObject* arg) {
    return py_append_vbr(self, arg, 4);
}

static PyObject* BitstreamWriter_vbr5(PyObject* self, PyObject* arg) {
    return py_append_vbr(self, arg, 5);
}

static PyObject* BitstreamWriter_vbr6(PyObject* self, PyObject* arg) {
    return py_append_vbr(self, arg, 6);
}

static PyObject* BitstreamWriter_vbr8(PyObject* self, PyObject* arg) {
    return py_append_vbr(self, arg, 8);
}

static void align32(BitstreamWriter* w) {
    if (w->buf_size) {
        w->words.push_back(static_cast<uint32_t>(w->buf));
        if (w->buf_size > 32)
            w->words.push_back(static_cast<uint32_t>(w->buf >> 32));
        w->buf_size = 0;
        w->buf = 0;
    }
}

static PyObject* BitstreamWriter_align_to_word(PyObject* self, PyObject* dummy) {
    BitstreamWriter& w = py_unwrap<BitstreamWriter>(self);
    align32(&w);
    return PyLong_FromSize_t(w.words.size());
}

static PyObject* BitstreamWriter_aligned_word(PyObject* self, PyObject* arg) {
    uint32_t value = pylong_as<uint32_t>(arg);
    if (PyErr_Occurred()) return nullptr;

    BitstreamWriter& w = py_unwrap<BitstreamWriter>(self);
    align32(&w);
    size_t pos = w.words.size();
    w.words.push_back(value);
    return PyLong_FromSize_t(pos);
}

static PyObject* BitstreamWriter_patch_word(PyObject* self,
                                            PyObject* const* args, Py_ssize_t nargs) {
    if (nargs != 2) {
        raise(PyExc_TypeError, "patch_word(offset, word) expects exactly 2 arguments, got ", nargs);
        return nullptr;
    }
    size_t offset = pylong_as<size_t>(args[0]);
    uint32_t word = pylong_as<uint32_t>(args[1]);
    if (PyErr_Occurred())
        return nullptr;
    BitstreamWriter& w = py_unwrap<BitstreamWriter>(self);
    if (offset >= w.words.size()) {
        raise(PyExc_IndexError,
              "Offset ", offset, " is out of range for stream of ", w.words.size(), " words");
        return nullptr;
    }
    w.words[offset] = word;
    return Py_NewRef(Py_None);
}

static PyObject* BitstreamWriter_raw_blob(PyObject* self, PyObject* blob) {
    if (!PyByteArray_Check(blob)) {
        raise(PyExc_TypeError, "Expected a bytearray argument");
        return nullptr;
    }

    BitstreamWriter& w = py_unwrap<BitstreamWriter>(self);
    align32(&w);
    const char* data = PyByteArray_AS_STRING(blob);
    size_t size = PyByteArray_GET_SIZE(blob);
    size_t num_words = (size + 3) / 4;
    size_t pos = w.words.size();
    w.words.resize(w.words.size() + num_words);
    mem_copy(w.words.data() + pos, data, size);
    return Py_NewRef(Py_None);
}

static PyObject* BitstreamWriter_to_bytes(PyObject* self, PyObject* dummy) {
    BitstreamWriter& w = py_unwrap<BitstreamWriter>(self);
    unsigned extra_words = (w.buf_size + 31) / 32;
    uint64_t buf = w.buf;
    for (unsigned i = 0; i < extra_words; ++i) {
        w.words.push_back(static_cast<uint32_t>(buf));
        buf >>= 32;
    }
    PyObject* ret = PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(w.words.data()), w.words.size() * 4);
    for (unsigned i = 0; i < extra_words; ++i)
        w.words.pop_back();
    return ret;
}

static PyMethodDef BitstreamWriter_methods[] = {
    {"fixed1", BitstreamWriter_fixed1, METH_O, nullptr},
    {"fixed2", BitstreamWriter_fixed2, METH_O, nullptr},
    {"fixed3", BitstreamWriter_fixed3, METH_O, nullptr},
    {"fixed4", BitstreamWriter_fixed4, METH_O, nullptr},
    {"fixed8", BitstreamWriter_fixed8, METH_O, nullptr},
    {"fixed32", BitstreamWriter_fixed32, METH_O, nullptr},
    {"vbr4", BitstreamWriter_vbr4, METH_O, nullptr},
    {"vbr5", BitstreamWriter_vbr5, METH_O, nullptr},
    {"vbr6", BitstreamWriter_vbr6, METH_O, nullptr},
    {"vbr8", BitstreamWriter_vbr8, METH_O, nullptr},
    {"align_to_word", BitstreamWriter_align_to_word, METH_NOARGS, nullptr},
    {"aligned_word", BitstreamWriter_aligned_word, METH_O, nullptr},
    {"patch_word", reinterpret_cast<PyCFunction>(BitstreamWriter_patch_word),
     METH_FASTCALL, nullptr},
    {"raw_blob", BitstreamWriter_raw_blob, METH_O, nullptr},
    {"to_bytes", BitstreamWriter_to_bytes, METH_NOARGS, nullptr},
    {}
};

PyTypeObject BitstreamWriter::pytype = {
    .tp_name = "cuda.tile._cext.BitstreamWriter",
    .tp_basicsize = sizeof(PythonWrapper<BitstreamWriter>),
    .tp_dealloc = pywrapper_dealloc<BitstreamWriter>,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_methods = BitstreamWriter_methods,
    .tp_new = pywrapper_new<BitstreamWriter>,
};


Status bitstream_init(PyObject* m) {
    if (PyType_Ready(&BitstreamWriter::pytype) < 0)
        return ErrorRaised;

    if (PyModule_AddObjectRef(m, "BitstreamWriter",
                reinterpret_cast<PyObject*>(&BitstreamWriter::pytype)) < 0)
        return ErrorRaised;

    return OK;
}


