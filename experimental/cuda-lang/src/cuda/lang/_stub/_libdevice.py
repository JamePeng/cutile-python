
# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from cuda.lang._stub._nvvm_support import (
    libdevice_function_stub as F,
    F32,
    F64,
    I16,
    I32,
    I64,
    P0
)


@F
def abs(x_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/abs.html#abs>`__.
    '''

__nv_abs = abs

@F
def acos(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/acos.html#acos>`__.
    '''

__nv_acos = acos

@F
def acosf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/acosf.html#acosf>`__.
    '''

__nv_acosf = acosf

@F
def acosh(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/acosh.html#acosh>`__.
    '''

__nv_acosh = acosh

@F
def acoshf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/acoshf.html#acoshf>`__.
    '''

__nv_acoshf = acoshf

@F
def asin(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/asin.html#asin>`__.
    '''

__nv_asin = asin

@F
def asinf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/asinf.html#asinf>`__.
    '''

__nv_asinf = asinf

@F
def asinh(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/asinh.html#asinh>`__.
    '''

__nv_asinh = asinh

@F
def asinhf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/asinhf.html#asinhf>`__.
    '''

__nv_asinhf = asinhf

@F
def atan(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/atan.html#atan>`__.
    '''

__nv_atan = atan

@F
def atan2(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/atan2.html#atan2>`__.
    '''

__nv_atan2 = atan2

@F
def atan2f(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/atan2f.html#atan2f>`__.
    '''

__nv_atan2f = atan2f

@F
def atanf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/atanf.html#atanf>`__.
    '''

__nv_atanf = atanf

@F
def atanh(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/atanh.html#atanh>`__.
    '''

__nv_atanh = atanh

@F
def atanhf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/atanhf.html#atanhf>`__.
    '''

__nv_atanhf = atanhf

@F
def brev(x_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/brev.html#brev>`__.
    '''

__nv_brev = brev

@F
def brevll(x_: I64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/brevll.html#brevll>`__.
    '''

__nv_brevll = brevll

@F
def byte_perm(x_: I32, y_: I32, z_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/byte_perm.html#byte_perm>`__.
    '''

__nv_byte_perm = byte_perm

@F
def cbrt(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/cbrt.html#cbrt>`__.
    '''

__nv_cbrt = cbrt

@F
def cbrtf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/cbrtf.html#cbrtf>`__.
    '''

__nv_cbrtf = cbrtf

@F
def ceil(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ceil.html#ceil>`__.
    '''

__nv_ceil = ceil

@F
def ceilf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ceilf.html#ceilf>`__.
    '''

__nv_ceilf = ceilf

@F
def clz(x_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/clz.html#clz>`__.
    '''

__nv_clz = clz

@F
def clzll(x_: I64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/clzll.html#clzll>`__.
    '''

__nv_clzll = clzll

@F
def copysign(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/copysign.html#copysign>`__.
    '''

__nv_copysign = copysign

@F
def copysignf(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/copysignf.html#copysignf>`__.
    '''

__nv_copysignf = copysignf

@F
def cos(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/cos.html#cos>`__.
    '''

__nv_cos = cos

@F
def cosf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/cosf.html#cosf>`__.
    '''

__nv_cosf = cosf

@F
def cosh(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/cosh.html#cosh>`__.
    '''

__nv_cosh = cosh

@F
def coshf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/coshf.html#coshf>`__.
    '''

__nv_coshf = coshf

@F
def cospi(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/cospi.html#cospi>`__.
    '''

__nv_cospi = cospi

@F
def cospif(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/cospif.html#cospif>`__.
    '''

__nv_cospif = cospif

@F
def dadd_rd(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/dadd_rd.html#dadd_rd>`__.
    '''

__nv_dadd_rd = dadd_rd

@F
def dadd_rn(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/dadd_rn.html#dadd_rn>`__.
    '''

__nv_dadd_rn = dadd_rn

@F
def dadd_ru(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/dadd_ru.html#dadd_ru>`__.
    '''

__nv_dadd_ru = dadd_ru

@F
def dadd_rz(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/dadd_rz.html#dadd_rz>`__.
    '''

__nv_dadd_rz = dadd_rz

@F
def ddiv_rd(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ddiv_rd.html#ddiv_rd>`__.
    '''

__nv_ddiv_rd = ddiv_rd

@F
def ddiv_rn(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ddiv_rn.html#ddiv_rn>`__.
    '''

__nv_ddiv_rn = ddiv_rn

@F
def ddiv_ru(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ddiv_ru.html#ddiv_ru>`__.
    '''

__nv_ddiv_ru = ddiv_ru

@F
def ddiv_rz(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ddiv_rz.html#ddiv_rz>`__.
    '''

__nv_ddiv_rz = ddiv_rz

@F
def dmul_rd(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/dmul_rd.html#dmul_rd>`__.
    '''

__nv_dmul_rd = dmul_rd

@F
def dmul_rn(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/dmul_rn.html#dmul_rn>`__.
    '''

__nv_dmul_rn = dmul_rn

@F
def dmul_ru(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/dmul_ru.html#dmul_ru>`__.
    '''

__nv_dmul_ru = dmul_ru

@F
def dmul_rz(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/dmul_rz.html#dmul_rz>`__.
    '''

__nv_dmul_rz = dmul_rz

@F
def double2float_rd(d_: F64) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2float_rd.html#double2float_rd>`__.
    '''

__nv_double2float_rd = double2float_rd

@F
def double2float_rn(d_: F64) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2float_rn.html#double2float_rn>`__.
    '''

__nv_double2float_rn = double2float_rn

@F
def double2float_ru(d_: F64) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2float_ru.html#double2float_ru>`__.
    '''

__nv_double2float_ru = double2float_ru

@F
def double2float_rz(d_: F64) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2float_rz.html#double2float_rz>`__.
    '''

__nv_double2float_rz = double2float_rz

@F
def double2hiint(d_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2hiint.html#double2hiint>`__.
    '''

__nv_double2hiint = double2hiint

@F
def double2int_rd(d_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2int_rd.html#double2int_rd>`__.
    '''

__nv_double2int_rd = double2int_rd

@F
def double2int_rn(d_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2int_rn.html#double2int_rn>`__.
    '''

__nv_double2int_rn = double2int_rn

@F
def double2int_ru(d_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2int_ru.html#double2int_ru>`__.
    '''

__nv_double2int_ru = double2int_ru

@F
def double2int_rz(d_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2int_rz.html#double2int_rz>`__.
    '''

__nv_double2int_rz = double2int_rz

@F
def double2ll_rd(f_: F64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2ll_rd.html#double2ll_rd>`__.
    '''

__nv_double2ll_rd = double2ll_rd

@F
def double2ll_rn(f_: F64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2ll_rn.html#double2ll_rn>`__.
    '''

__nv_double2ll_rn = double2ll_rn

@F
def double2ll_ru(f_: F64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2ll_ru.html#double2ll_ru>`__.
    '''

__nv_double2ll_ru = double2ll_ru

@F
def double2ll_rz(f_: F64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2ll_rz.html#double2ll_rz>`__.
    '''

__nv_double2ll_rz = double2ll_rz

@F
def double2loint(d_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2loint.html#double2loint>`__.
    '''

__nv_double2loint = double2loint

@F
def double2uint_rd(d_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2uint_rd.html#double2uint_rd>`__.
    '''

__nv_double2uint_rd = double2uint_rd

@F
def double2uint_rn(d_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2uint_rn.html#double2uint_rn>`__.
    '''

__nv_double2uint_rn = double2uint_rn

@F
def double2uint_ru(d_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2uint_ru.html#double2uint_ru>`__.
    '''

__nv_double2uint_ru = double2uint_ru

@F
def double2uint_rz(d_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2uint_rz.html#double2uint_rz>`__.
    '''

__nv_double2uint_rz = double2uint_rz

@F
def double2ull_rd(f_: F64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2ull_rd.html#double2ull_rd>`__.
    '''

__nv_double2ull_rd = double2ull_rd

@F
def double2ull_rn(f_: F64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2ull_rn.html#double2ull_rn>`__.
    '''

__nv_double2ull_rn = double2ull_rn

@F
def double2ull_ru(f_: F64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2ull_ru.html#double2ull_ru>`__.
    '''

__nv_double2ull_ru = double2ull_ru

@F
def double2ull_rz(f_: F64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double2ull_rz.html#double2ull_rz>`__.
    '''

__nv_double2ull_rz = double2ull_rz

@F
def double_as_longlong(x_: F64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/double_as_longlong.html#double_as_longlong>`__.
    '''

__nv_double_as_longlong = double_as_longlong

@F
def drcp_rd(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/drcp_rd.html#drcp_rd>`__.
    '''

__nv_drcp_rd = drcp_rd

@F
def drcp_rn(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/drcp_rn.html#drcp_rn>`__.
    '''

__nv_drcp_rn = drcp_rn

@F
def drcp_ru(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/drcp_ru.html#drcp_ru>`__.
    '''

__nv_drcp_ru = drcp_ru

@F
def drcp_rz(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/drcp_rz.html#drcp_rz>`__.
    '''

__nv_drcp_rz = drcp_rz

@F
def dsqrt_rd(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/dsqrt_rd.html#dsqrt_rd>`__.
    '''

__nv_dsqrt_rd = dsqrt_rd

@F
def dsqrt_rn(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/dsqrt_rn.html#dsqrt_rn>`__.
    '''

__nv_dsqrt_rn = dsqrt_rn

@F
def dsqrt_ru(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/dsqrt_ru.html#dsqrt_ru>`__.
    '''

__nv_dsqrt_ru = dsqrt_ru

@F
def dsqrt_rz(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/dsqrt_rz.html#dsqrt_rz>`__.
    '''

__nv_dsqrt_rz = dsqrt_rz

@F
def erf(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/erf.html#erf>`__.
    '''

__nv_erf = erf

@F
def erfc(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/erfc.html#erfc>`__.
    '''

__nv_erfc = erfc

@F
def erfcf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/erfcf.html#erfcf>`__.
    '''

__nv_erfcf = erfcf

@F
def erfcinv(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/erfcinv.html#erfcinv>`__.
    '''

__nv_erfcinv = erfcinv

@F
def erfcinvf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/erfcinvf.html#erfcinvf>`__.
    '''

__nv_erfcinvf = erfcinvf

@F
def erfcx(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/erfcx.html#erfcx>`__.
    '''

__nv_erfcx = erfcx

@F
def erfcxf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/erfcxf.html#erfcxf>`__.
    '''

__nv_erfcxf = erfcxf

@F
def erff(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/erff.html#erff>`__.
    '''

__nv_erff = erff

@F
def erfinv(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/erfinv.html#erfinv>`__.
    '''

__nv_erfinv = erfinv

@F
def erfinvf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/erfinvf.html#erfinvf>`__.
    '''

__nv_erfinvf = erfinvf

@F
def exp(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/exp.html#exp>`__.
    '''

__nv_exp = exp

@F
def exp10(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/exp10.html#exp10>`__.
    '''

__nv_exp10 = exp10

@F
def exp10f(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/exp10f.html#exp10f>`__.
    '''

__nv_exp10f = exp10f

@F
def exp2(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/exp2.html#exp2>`__.
    '''

__nv_exp2 = exp2

@F
def exp2f(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/exp2f.html#exp2f>`__.
    '''

__nv_exp2f = exp2f

@F
def expf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/expf.html#expf>`__.
    '''

__nv_expf = expf

@F
def expm1(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/expm1.html#expm1>`__.
    '''

__nv_expm1 = expm1

@F
def expm1f(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/expm1f.html#expm1f>`__.
    '''

__nv_expm1f = expm1f

@F
def fabs(f_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fabs.html#fabs>`__.
    '''

__nv_fabs = fabs

@F
def fabsf(f_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fabsf.html#fabsf>`__.
    '''

__nv_fabsf = fabsf

@F
def fadd_rd(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fadd_rd.html#fadd_rd>`__.
    '''

__nv_fadd_rd = fadd_rd

@F
def fadd_rn(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fadd_rn.html#fadd_rn>`__.
    '''

__nv_fadd_rn = fadd_rn

@F
def fadd_ru(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fadd_ru.html#fadd_ru>`__.
    '''

__nv_fadd_ru = fadd_ru

@F
def fadd_rz(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fadd_rz.html#fadd_rz>`__.
    '''

__nv_fadd_rz = fadd_rz

@F
def fast_cosf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fast_cosf.html#fast_cosf>`__.
    '''

__nv_fast_cosf = fast_cosf

@F
def fast_exp10f(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fast_exp10f.html#fast_exp10f>`__.
    '''

__nv_fast_exp10f = fast_exp10f

@F
def fast_expf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fast_expf.html#fast_expf>`__.
    '''

__nv_fast_expf = fast_expf

@F
def fast_fdividef(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fast_fdividef.html#fast_fdividef>`__.
    '''

__nv_fast_fdividef = fast_fdividef

@F
def fast_log10f(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fast_log10f.html#fast_log10f>`__.
    '''

__nv_fast_log10f = fast_log10f

@F
def fast_log2f(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fast_log2f.html#fast_log2f>`__.
    '''

__nv_fast_log2f = fast_log2f

@F
def fast_logf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fast_logf.html#fast_logf>`__.
    '''

__nv_fast_logf = fast_logf

@F
def fast_powf(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fast_powf.html#fast_powf>`__.
    '''

__nv_fast_powf = fast_powf

@F
def __nv_fast_sincosf(x_: F32, sptr_: P0, cptr_: P0) -> None:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_fast_sincosf.html#__nv_fast_sincosf>`__.
    '''

@F
def fast_sinf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fast_sinf.html#fast_sinf>`__.
    '''

__nv_fast_sinf = fast_sinf

@F
def fast_tanf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fast_tanf.html#fast_tanf>`__.
    '''

__nv_fast_tanf = fast_tanf

@F
def fdim(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fdim.html#fdim>`__.
    '''

__nv_fdim = fdim

@F
def fdimf(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fdimf.html#fdimf>`__.
    '''

__nv_fdimf = fdimf

@F
def fdiv_rd(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fdiv_rd.html#fdiv_rd>`__.
    '''

__nv_fdiv_rd = fdiv_rd

@F
def fdiv_rn(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fdiv_rn.html#fdiv_rn>`__.
    '''

__nv_fdiv_rn = fdiv_rn

@F
def fdiv_ru(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fdiv_ru.html#fdiv_ru>`__.
    '''

__nv_fdiv_ru = fdiv_ru

@F
def fdiv_rz(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fdiv_rz.html#fdiv_rz>`__.
    '''

__nv_fdiv_rz = fdiv_rz

@F
def ffs(x_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ffs.html#ffs>`__.
    '''

__nv_ffs = ffs

@F
def ffsll(x_: I64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ffsll.html#ffsll>`__.
    '''

__nv_ffsll = ffsll

@F
def finitef(x_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/finitef.html#finitef>`__.
    '''

__nv_finitef = finitef

@F
def float2half_rn(f_: F32) -> I16:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2half_rn.html#float2half_rn>`__.
    '''

__nv_float2half_rn = float2half_rn

@F
def float2int_rd(in_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2int_rd.html#float2int_rd>`__.
    '''

__nv_float2int_rd = float2int_rd

@F
def float2int_rn(in_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2int_rn.html#float2int_rn>`__.
    '''

__nv_float2int_rn = float2int_rn

@F
def float2int_ru(in_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2int_ru.html#float2int_ru>`__.
    '''

__nv_float2int_ru = float2int_ru

@F
def float2int_rz(in_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2int_rz.html#float2int_rz>`__.
    '''

__nv_float2int_rz = float2int_rz

@F
def float2ll_rd(f_: F32) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2ll_rd.html#float2ll_rd>`__.
    '''

__nv_float2ll_rd = float2ll_rd

@F
def float2ll_rn(f_: F32) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2ll_rn.html#float2ll_rn>`__.
    '''

__nv_float2ll_rn = float2ll_rn

@F
def float2ll_ru(f_: F32) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2ll_ru.html#float2ll_ru>`__.
    '''

__nv_float2ll_ru = float2ll_ru

@F
def float2ll_rz(f_: F32) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2ll_rz.html#float2ll_rz>`__.
    '''

__nv_float2ll_rz = float2ll_rz

@F
def float2uint_rd(in_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2uint_rd.html#float2uint_rd>`__.
    '''

__nv_float2uint_rd = float2uint_rd

@F
def float2uint_rn(in_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2uint_rn.html#float2uint_rn>`__.
    '''

__nv_float2uint_rn = float2uint_rn

@F
def float2uint_ru(in_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2uint_ru.html#float2uint_ru>`__.
    '''

__nv_float2uint_ru = float2uint_ru

@F
def float2uint_rz(in_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2uint_rz.html#float2uint_rz>`__.
    '''

__nv_float2uint_rz = float2uint_rz

@F
def float2ull_rd(f_: F32) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2ull_rd.html#float2ull_rd>`__.
    '''

__nv_float2ull_rd = float2ull_rd

@F
def float2ull_rn(f_: F32) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2ull_rn.html#float2ull_rn>`__.
    '''

__nv_float2ull_rn = float2ull_rn

@F
def float2ull_ru(f_: F32) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2ull_ru.html#float2ull_ru>`__.
    '''

__nv_float2ull_ru = float2ull_ru

@F
def float2ull_rz(f_: F32) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float2ull_rz.html#float2ull_rz>`__.
    '''

__nv_float2ull_rz = float2ull_rz

@F
def float_as_int(x_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/float_as_int.html#float_as_int>`__.
    '''

__nv_float_as_int = float_as_int

@F
def floor(f_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/floor.html#floor>`__.
    '''

__nv_floor = floor

@F
def floorf(f_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/floorf.html#floorf>`__.
    '''

__nv_floorf = floorf

@F
def fma(x_: F64, y_: F64, z_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fma.html#fma>`__.
    '''

__nv_fma = fma

@F
def fma_rd(x_: F64, y_: F64, z_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fma_rd.html#fma_rd>`__.
    '''

__nv_fma_rd = fma_rd

@F
def fma_rn(x_: F64, y_: F64, z_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fma_rn.html#fma_rn>`__.
    '''

__nv_fma_rn = fma_rn

@F
def fma_ru(x_: F64, y_: F64, z_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fma_ru.html#fma_ru>`__.
    '''

__nv_fma_ru = fma_ru

@F
def fma_rz(x_: F64, y_: F64, z_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fma_rz.html#fma_rz>`__.
    '''

__nv_fma_rz = fma_rz

@F
def fmaf(x_: F32, y_: F32, z_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmaf.html#fmaf>`__.
    '''

__nv_fmaf = fmaf

@F
def fmaf_rd(x_: F32, y_: F32, z_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmaf_rd.html#fmaf_rd>`__.
    '''

__nv_fmaf_rd = fmaf_rd

@F
def fmaf_rn(x_: F32, y_: F32, z_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmaf_rn.html#fmaf_rn>`__.
    '''

__nv_fmaf_rn = fmaf_rn

@F
def fmaf_ru(x_: F32, y_: F32, z_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmaf_ru.html#fmaf_ru>`__.
    '''

__nv_fmaf_ru = fmaf_ru

@F
def fmaf_rz(x_: F32, y_: F32, z_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmaf_rz.html#fmaf_rz>`__.
    '''

__nv_fmaf_rz = fmaf_rz

@F
def fmax(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmax.html#fmax>`__.
    '''

__nv_fmax = fmax

@F
def fmaxf(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmaxf.html#fmaxf>`__.
    '''

__nv_fmaxf = fmaxf

@F
def fmin(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmin.html#fmin>`__.
    '''

__nv_fmin = fmin

@F
def fminf(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fminf.html#fminf>`__.
    '''

__nv_fminf = fminf

@F
def fmod(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmod.html#fmod>`__.
    '''

__nv_fmod = fmod

@F
def fmodf(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmodf.html#fmodf>`__.
    '''

__nv_fmodf = fmodf

@F
def fmul_rd(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmul_rd.html#fmul_rd>`__.
    '''

__nv_fmul_rd = fmul_rd

@F
def fmul_rn(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmul_rn.html#fmul_rn>`__.
    '''

__nv_fmul_rn = fmul_rn

@F
def fmul_ru(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmul_ru.html#fmul_ru>`__.
    '''

__nv_fmul_ru = fmul_ru

@F
def fmul_rz(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fmul_rz.html#fmul_rz>`__.
    '''

__nv_fmul_rz = fmul_rz

@F
def frcp_rd(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/frcp_rd.html#frcp_rd>`__.
    '''

__nv_frcp_rd = frcp_rd

@F
def frcp_rn(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/frcp_rn.html#frcp_rn>`__.
    '''

__nv_frcp_rn = frcp_rn

@F
def frcp_ru(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/frcp_ru.html#frcp_ru>`__.
    '''

__nv_frcp_ru = frcp_ru

@F
def frcp_rz(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/frcp_rz.html#frcp_rz>`__.
    '''

__nv_frcp_rz = frcp_rz

@F
def __nv_frexp(x_: F64, b_: P0) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frexp.html#__nv_frexp>`__.
    '''

@F
def __nv_frexpf(x_: F32, b_: P0) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_frexpf.html#__nv_frexpf>`__.
    '''

@F
def frsqrt_rn(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/frsqrt_rn.html#frsqrt_rn>`__.
    '''

__nv_frsqrt_rn = frsqrt_rn

@F
def fsqrt_rd(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fsqrt_rd.html#fsqrt_rd>`__.
    '''

__nv_fsqrt_rd = fsqrt_rd

@F
def fsqrt_rn(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fsqrt_rn.html#fsqrt_rn>`__.
    '''

__nv_fsqrt_rn = fsqrt_rn

@F
def fsqrt_ru(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fsqrt_ru.html#fsqrt_ru>`__.
    '''

__nv_fsqrt_ru = fsqrt_ru

@F
def fsqrt_rz(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fsqrt_rz.html#fsqrt_rz>`__.
    '''

__nv_fsqrt_rz = fsqrt_rz

@F
def fsub_rd(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fsub_rd.html#fsub_rd>`__.
    '''

__nv_fsub_rd = fsub_rd

@F
def fsub_rn(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fsub_rn.html#fsub_rn>`__.
    '''

__nv_fsub_rn = fsub_rn

@F
def fsub_ru(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fsub_ru.html#fsub_ru>`__.
    '''

__nv_fsub_ru = fsub_ru

@F
def fsub_rz(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/fsub_rz.html#fsub_rz>`__.
    '''

__nv_fsub_rz = fsub_rz

@F
def hadd(x_: I32, y_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/hadd.html#hadd>`__.
    '''

__nv_hadd = hadd

@F
def half2float(h_: I16) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/half2float.html#half2float>`__.
    '''

__nv_half2float = half2float

@F
def hiloint2double(x_: I32, y_: I32) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/hiloint2double.html#hiloint2double>`__.
    '''

__nv_hiloint2double = hiloint2double

@F
def hypot(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/hypot.html#hypot>`__.
    '''

__nv_hypot = hypot

@F
def hypotf(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/hypotf.html#hypotf>`__.
    '''

__nv_hypotf = hypotf

@F
def ilogb(x_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ilogb.html#ilogb>`__.
    '''

__nv_ilogb = ilogb

@F
def ilogbf(x_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ilogbf.html#ilogbf>`__.
    '''

__nv_ilogbf = ilogbf

@F
def int2double_rn(i_: I32) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/int2double_rn.html#int2double_rn>`__.
    '''

__nv_int2double_rn = int2double_rn

@F
def int2float_rd(in_: I32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/int2float_rd.html#int2float_rd>`__.
    '''

__nv_int2float_rd = int2float_rd

@F
def int2float_rn(in_: I32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/int2float_rn.html#int2float_rn>`__.
    '''

__nv_int2float_rn = int2float_rn

@F
def int2float_ru(in_: I32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/int2float_ru.html#int2float_ru>`__.
    '''

__nv_int2float_ru = int2float_ru

@F
def int2float_rz(in_: I32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/int2float_rz.html#int2float_rz>`__.
    '''

__nv_int2float_rz = int2float_rz

@F
def int_as_float(x_: I32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/int_as_float.html#int_as_float>`__.
    '''

__nv_int_as_float = int_as_float

@F
def isfinited(x_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/isfinited.html#isfinited>`__.
    '''

__nv_isfinited = isfinited

@F
def isinfd(x_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/isinfd.html#isinfd>`__.
    '''

__nv_isinfd = isinfd

@F
def isinff(x_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/isinff.html#isinff>`__.
    '''

__nv_isinff = isinff

@F
def isnand(x_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/isnand.html#isnand>`__.
    '''

__nv_isnand = isnand

@F
def isnanf(x_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/isnanf.html#isnanf>`__.
    '''

__nv_isnanf = isnanf

@F
def j0(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/j0.html#j0>`__.
    '''

__nv_j0 = j0

@F
def j0f(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/j0f.html#j0f>`__.
    '''

__nv_j0f = j0f

@F
def j1(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/j1.html#j1>`__.
    '''

__nv_j1 = j1

@F
def j1f(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/j1f.html#j1f>`__.
    '''

__nv_j1f = j1f

@F
def jn(n_: I32, x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/jn.html#jn>`__.
    '''

__nv_jn = jn

@F
def jnf(n_: I32, x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/jnf.html#jnf>`__.
    '''

__nv_jnf = jnf

@F
def ldexp(x_: F64, y_: I32) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ldexp.html#ldexp>`__.
    '''

__nv_ldexp = ldexp

@F
def ldexpf(x_: F32, y_: I32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ldexpf.html#ldexpf>`__.
    '''

__nv_ldexpf = ldexpf

@F
def lgamma(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/lgamma.html#lgamma>`__.
    '''

__nv_lgamma = lgamma

@F
def lgammaf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/lgammaf.html#lgammaf>`__.
    '''

__nv_lgammaf = lgammaf

@F
def ll2double_rd(l_: I64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ll2double_rd.html#ll2double_rd>`__.
    '''

__nv_ll2double_rd = ll2double_rd

@F
def ll2double_rn(l_: I64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ll2double_rn.html#ll2double_rn>`__.
    '''

__nv_ll2double_rn = ll2double_rn

@F
def ll2double_ru(l_: I64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ll2double_ru.html#ll2double_ru>`__.
    '''

__nv_ll2double_ru = ll2double_ru

@F
def ll2double_rz(l_: I64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ll2double_rz.html#ll2double_rz>`__.
    '''

__nv_ll2double_rz = ll2double_rz

@F
def ll2float_rd(l_: I64) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ll2float_rd.html#ll2float_rd>`__.
    '''

__nv_ll2float_rd = ll2float_rd

@F
def ll2float_rn(l_: I64) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ll2float_rn.html#ll2float_rn>`__.
    '''

__nv_ll2float_rn = ll2float_rn

@F
def ll2float_ru(l_: I64) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ll2float_ru.html#ll2float_ru>`__.
    '''

__nv_ll2float_ru = ll2float_ru

@F
def ll2float_rz(l_: I64) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ll2float_rz.html#ll2float_rz>`__.
    '''

__nv_ll2float_rz = ll2float_rz

@F
def llabs(x_: I64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/llabs.html#llabs>`__.
    '''

__nv_llabs = llabs

@F
def llmax(x_: I64, y_: I64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/llmax.html#llmax>`__.
    '''

__nv_llmax = llmax

@F
def llmin(x_: I64, y_: I64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/llmin.html#llmin>`__.
    '''

__nv_llmin = llmin

@F
def llrint(x_: F64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/llrint.html#llrint>`__.
    '''

__nv_llrint = llrint

@F
def llrintf(x_: F32) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/llrintf.html#llrintf>`__.
    '''

__nv_llrintf = llrintf

@F
def llround(x_: F64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/llround.html#llround>`__.
    '''

__nv_llround = llround

@F
def llroundf(x_: F32) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/llroundf.html#llroundf>`__.
    '''

__nv_llroundf = llroundf

@F
def log(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/log.html#log>`__.
    '''

__nv_log = log

@F
def log10(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/log10.html#log10>`__.
    '''

__nv_log10 = log10

@F
def log10f(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/log10f.html#log10f>`__.
    '''

__nv_log10f = log10f

@F
def log1p(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/log1p.html#log1p>`__.
    '''

__nv_log1p = log1p

@F
def log1pf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/log1pf.html#log1pf>`__.
    '''

__nv_log1pf = log1pf

@F
def log2(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/log2.html#log2>`__.
    '''

__nv_log2 = log2

@F
def log2f(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/log2f.html#log2f>`__.
    '''

__nv_log2f = log2f

@F
def logb(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/logb.html#logb>`__.
    '''

__nv_logb = logb

@F
def logbf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/logbf.html#logbf>`__.
    '''

__nv_logbf = logbf

@F
def logf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/logf.html#logf>`__.
    '''

__nv_logf = logf

@F
def longlong_as_double(x_: I64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/longlong_as_double.html#longlong_as_double>`__.
    '''

__nv_longlong_as_double = longlong_as_double

@F
def max(x_: I32, y_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/max.html#max>`__.
    '''

__nv_max = max

@F
def min(x_: I32, y_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/min.html#min>`__.
    '''

__nv_min = min

@F
def __nv_modf(x_: F64, b_: P0) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_modf.html#__nv_modf>`__.
    '''

@F
def __nv_modff(x_: F32, b_: P0) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_modff.html#__nv_modff>`__.
    '''

@F
def mul24(x_: I32, y_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/mul24.html#mul24>`__.
    '''

__nv_mul24 = mul24

@F
def mul64hi(x_: I64, y_: I64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/mul64hi.html#mul64hi>`__.
    '''

__nv_mul64hi = mul64hi

@F
def mulhi(x_: I32, y_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/mulhi.html#mulhi>`__.
    '''

__nv_mulhi = mulhi

@F
def nearbyint(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/nearbyint.html#nearbyint>`__.
    '''

__nv_nearbyint = nearbyint

@F
def nearbyintf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/nearbyintf.html#nearbyintf>`__.
    '''

__nv_nearbyintf = nearbyintf

@F
def nextafter(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/nextafter.html#nextafter>`__.
    '''

__nv_nextafter = nextafter

@F
def nextafterf(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/nextafterf.html#nextafterf>`__.
    '''

__nv_nextafterf = nextafterf

@F
def normcdf(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/normcdf.html#normcdf>`__.
    '''

__nv_normcdf = normcdf

@F
def normcdff(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/normcdff.html#normcdff>`__.
    '''

__nv_normcdff = normcdff

@F
def normcdfinv(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/normcdfinv.html#normcdfinv>`__.
    '''

__nv_normcdfinv = normcdfinv

@F
def normcdfinvf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/normcdfinvf.html#normcdfinvf>`__.
    '''

__nv_normcdfinvf = normcdfinvf

@F
def popc(x_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/popc.html#popc>`__.
    '''

__nv_popc = popc

@F
def popcll(x_: I64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/popcll.html#popcll>`__.
    '''

__nv_popcll = popcll

@F
def pow(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/pow.html#pow>`__.
    '''

__nv_pow = pow

@F
def powf(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/powf.html#powf>`__.
    '''

__nv_powf = powf

@F
def powi(x_: F64, y_: I32) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/powi.html#powi>`__.
    '''

__nv_powi = powi

@F
def powif(x_: F32, y_: I32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/powif.html#powif>`__.
    '''

__nv_powif = powif

@F
def rcbrt(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/rcbrt.html#rcbrt>`__.
    '''

__nv_rcbrt = rcbrt

@F
def rcbrtf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/rcbrtf.html#rcbrtf>`__.
    '''

__nv_rcbrtf = rcbrtf

@F
def remainder(x_: F64, y_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/remainder.html#remainder>`__.
    '''

__nv_remainder = remainder

@F
def remainderf(x_: F32, y_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/remainderf.html#remainderf>`__.
    '''

__nv_remainderf = remainderf

@F
def __nv_remquo(x_: F64, y_: F64, c_: P0) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remquo.html#__nv_remquo>`__.
    '''

@F
def __nv_remquof(x_: F32, y_: F32, quo_: P0) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_remquof.html#__nv_remquof>`__.
    '''

@F
def rhadd(x_: I32, y_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/rhadd.html#rhadd>`__.
    '''

__nv_rhadd = rhadd

@F
def rint(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/rint.html#rint>`__.
    '''

__nv_rint = rint

@F
def rintf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/rintf.html#rintf>`__.
    '''

__nv_rintf = rintf

@F
def round(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/round.html#round>`__.
    '''

__nv_round = round

@F
def roundf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/roundf.html#roundf>`__.
    '''

__nv_roundf = roundf

@F
def rsqrt(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/rsqrt.html#rsqrt>`__.
    '''

__nv_rsqrt = rsqrt

@F
def rsqrtf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/rsqrtf.html#rsqrtf>`__.
    '''

__nv_rsqrtf = rsqrtf

@F
def sad(x_: I32, y_: I32, z_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/sad.html#sad>`__.
    '''

__nv_sad = sad

@F
def saturatef(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/saturatef.html#saturatef>`__.
    '''

__nv_saturatef = saturatef

@F
def scalbn(x_: F64, y_: I32) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/scalbn.html#scalbn>`__.
    '''

__nv_scalbn = scalbn

@F
def scalbnf(x_: F32, y_: I32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/scalbnf.html#scalbnf>`__.
    '''

__nv_scalbnf = scalbnf

@F
def signbitd(x_: F64) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/signbitd.html#signbitd>`__.
    '''

__nv_signbitd = signbitd

@F
def signbitf(x_: F32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/signbitf.html#signbitf>`__.
    '''

__nv_signbitf = signbitf

@F
def sin(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/sin.html#sin>`__.
    '''

__nv_sin = sin

@F
def __nv_sincos(x_: F64, sptr_: P0, cptr_: P0) -> None:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincos.html#__nv_sincos>`__.
    '''

@F
def __nv_sincosf(x_: F32, sptr_: P0, cptr_: P0) -> None:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincosf.html#__nv_sincosf>`__.
    '''

@F
def __nv_sincospi(x_: F64, sptr_: P0, cptr_: P0) -> None:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincospi.html#__nv_sincospi>`__.
    '''

@F
def __nv_sincospif(x_: F32, sptr_: P0, cptr_: P0) -> None:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_sincospif.html#__nv_sincospif>`__.
    '''

@F
def sinf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/sinf.html#sinf>`__.
    '''

__nv_sinf = sinf

@F
def sinh(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/sinh.html#sinh>`__.
    '''

__nv_sinh = sinh

@F
def sinhf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/sinhf.html#sinhf>`__.
    '''

__nv_sinhf = sinhf

@F
def sinpi(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/sinpi.html#sinpi>`__.
    '''

__nv_sinpi = sinpi

@F
def sinpif(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/sinpif.html#sinpif>`__.
    '''

__nv_sinpif = sinpif

@F
def sqrt(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/sqrt.html#sqrt>`__.
    '''

__nv_sqrt = sqrt

@F
def sqrtf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/sqrtf.html#sqrtf>`__.
    '''

__nv_sqrtf = sqrtf

@F
def tan(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/tan.html#tan>`__.
    '''

__nv_tan = tan

@F
def tanf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/tanf.html#tanf>`__.
    '''

__nv_tanf = tanf

@F
def tanh(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/tanh.html#tanh>`__.
    '''

__nv_tanh = tanh

@F
def tanhf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/tanhf.html#tanhf>`__.
    '''

__nv_tanhf = tanhf

@F
def tgamma(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/tgamma.html#tgamma>`__.
    '''

__nv_tgamma = tgamma

@F
def tgammaf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/tgammaf.html#tgammaf>`__.
    '''

__nv_tgammaf = tgammaf

@F
def trunc(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/trunc.html#trunc>`__.
    '''

__nv_trunc = trunc

@F
def truncf(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/truncf.html#truncf>`__.
    '''

__nv_truncf = truncf

@F
def uhadd(x_: I32, y_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/uhadd.html#uhadd>`__.
    '''

__nv_uhadd = uhadd

@F
def uint2double_rn(i_: I32) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/uint2double_rn.html#uint2double_rn>`__.
    '''

__nv_uint2double_rn = uint2double_rn

@F
def uint2float_rd(in_: I32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/uint2float_rd.html#uint2float_rd>`__.
    '''

__nv_uint2float_rd = uint2float_rd

@F
def uint2float_rn(in_: I32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/uint2float_rn.html#uint2float_rn>`__.
    '''

__nv_uint2float_rn = uint2float_rn

@F
def uint2float_ru(in_: I32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/uint2float_ru.html#uint2float_ru>`__.
    '''

__nv_uint2float_ru = uint2float_ru

@F
def uint2float_rz(in_: I32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/uint2float_rz.html#uint2float_rz>`__.
    '''

__nv_uint2float_rz = uint2float_rz

@F
def ull2double_rd(l_: I64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ull2double_rd.html#ull2double_rd>`__.
    '''

__nv_ull2double_rd = ull2double_rd

@F
def ull2double_rn(l_: I64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ull2double_rn.html#ull2double_rn>`__.
    '''

__nv_ull2double_rn = ull2double_rn

@F
def ull2double_ru(l_: I64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ull2double_ru.html#ull2double_ru>`__.
    '''

__nv_ull2double_ru = ull2double_ru

@F
def ull2double_rz(l_: I64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ull2double_rz.html#ull2double_rz>`__.
    '''

__nv_ull2double_rz = ull2double_rz

@F
def ull2float_rd(l_: I64) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ull2float_rd.html#ull2float_rd>`__.
    '''

__nv_ull2float_rd = ull2float_rd

@F
def ull2float_rn(l_: I64) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ull2float_rn.html#ull2float_rn>`__.
    '''

__nv_ull2float_rn = ull2float_rn

@F
def ull2float_ru(l_: I64) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ull2float_ru.html#ull2float_ru>`__.
    '''

__nv_ull2float_ru = ull2float_ru

@F
def ull2float_rz(l_: I64) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ull2float_rz.html#ull2float_rz>`__.
    '''

__nv_ull2float_rz = ull2float_rz

@F
def ullmax(x_: I64, y_: I64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ullmax.html#ullmax>`__.
    '''

__nv_ullmax = ullmax

@F
def ullmin(x_: I64, y_: I64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ullmin.html#ullmin>`__.
    '''

__nv_ullmin = ullmin

@F
def umax(x_: I32, y_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/umax.html#umax>`__.
    '''

__nv_umax = umax

@F
def umin(x_: I32, y_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/umin.html#umin>`__.
    '''

__nv_umin = umin

@F
def umul24(x_: I32, y_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/umul24.html#umul24>`__.
    '''

__nv_umul24 = umul24

@F
def umul64hi(x_: I64, y_: I64) -> I64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/umul64hi.html#umul64hi>`__.
    '''

__nv_umul64hi = umul64hi

@F
def umulhi(x_: I32, y_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/umulhi.html#umulhi>`__.
    '''

__nv_umulhi = umulhi

@F
def urhadd(x_: I32, y_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/urhadd.html#urhadd>`__.
    '''

__nv_urhadd = urhadd

@F
def usad(x_: I32, y_: I32, z_: I32) -> I32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/usad.html#usad>`__.
    '''

__nv_usad = usad

@F
def y0(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/y0.html#y0>`__.
    '''

__nv_y0 = y0

@F
def y0f(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/y0f.html#y0f>`__.
    '''

__nv_y0f = y0f

@F
def y1(x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/y1.html#y1>`__.
    '''

__nv_y1 = y1

@F
def y1f(x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/y1f.html#y1f>`__.
    '''

__nv_y1f = y1f

@F
def yn(n_: I32, x_: F64) -> F64:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/yn.html#yn>`__.
    '''

__nv_yn = yn

@F
def ynf(n_: I32, x_: F32) -> F32:
    '''
    See `official documentation <https://docs.nvidia.com/cuda/libdevice-users-guide/ynf.html#ynf>`__.
    '''

__nv_ynf = ynf
__all__ = (
    "abs",
    "__nv_abs",
    "acos",
    "__nv_acos",
    "acosf",
    "__nv_acosf",
    "acosh",
    "__nv_acosh",
    "acoshf",
    "__nv_acoshf",
    "asin",
    "__nv_asin",
    "asinf",
    "__nv_asinf",
    "asinh",
    "__nv_asinh",
    "asinhf",
    "__nv_asinhf",
    "atan",
    "__nv_atan",
    "atan2",
    "__nv_atan2",
    "atan2f",
    "__nv_atan2f",
    "atanf",
    "__nv_atanf",
    "atanh",
    "__nv_atanh",
    "atanhf",
    "__nv_atanhf",
    "brev",
    "__nv_brev",
    "brevll",
    "__nv_brevll",
    "byte_perm",
    "__nv_byte_perm",
    "cbrt",
    "__nv_cbrt",
    "cbrtf",
    "__nv_cbrtf",
    "ceil",
    "__nv_ceil",
    "ceilf",
    "__nv_ceilf",
    "clz",
    "__nv_clz",
    "clzll",
    "__nv_clzll",
    "copysign",
    "__nv_copysign",
    "copysignf",
    "__nv_copysignf",
    "cos",
    "__nv_cos",
    "cosf",
    "__nv_cosf",
    "cosh",
    "__nv_cosh",
    "coshf",
    "__nv_coshf",
    "cospi",
    "__nv_cospi",
    "cospif",
    "__nv_cospif",
    "dadd_rd",
    "__nv_dadd_rd",
    "dadd_rn",
    "__nv_dadd_rn",
    "dadd_ru",
    "__nv_dadd_ru",
    "dadd_rz",
    "__nv_dadd_rz",
    "ddiv_rd",
    "__nv_ddiv_rd",
    "ddiv_rn",
    "__nv_ddiv_rn",
    "ddiv_ru",
    "__nv_ddiv_ru",
    "ddiv_rz",
    "__nv_ddiv_rz",
    "dmul_rd",
    "__nv_dmul_rd",
    "dmul_rn",
    "__nv_dmul_rn",
    "dmul_ru",
    "__nv_dmul_ru",
    "dmul_rz",
    "__nv_dmul_rz",
    "double2float_rd",
    "__nv_double2float_rd",
    "double2float_rn",
    "__nv_double2float_rn",
    "double2float_ru",
    "__nv_double2float_ru",
    "double2float_rz",
    "__nv_double2float_rz",
    "double2hiint",
    "__nv_double2hiint",
    "double2int_rd",
    "__nv_double2int_rd",
    "double2int_rn",
    "__nv_double2int_rn",
    "double2int_ru",
    "__nv_double2int_ru",
    "double2int_rz",
    "__nv_double2int_rz",
    "double2ll_rd",
    "__nv_double2ll_rd",
    "double2ll_rn",
    "__nv_double2ll_rn",
    "double2ll_ru",
    "__nv_double2ll_ru",
    "double2ll_rz",
    "__nv_double2ll_rz",
    "double2loint",
    "__nv_double2loint",
    "double2uint_rd",
    "__nv_double2uint_rd",
    "double2uint_rn",
    "__nv_double2uint_rn",
    "double2uint_ru",
    "__nv_double2uint_ru",
    "double2uint_rz",
    "__nv_double2uint_rz",
    "double2ull_rd",
    "__nv_double2ull_rd",
    "double2ull_rn",
    "__nv_double2ull_rn",
    "double2ull_ru",
    "__nv_double2ull_ru",
    "double2ull_rz",
    "__nv_double2ull_rz",
    "double_as_longlong",
    "__nv_double_as_longlong",
    "drcp_rd",
    "__nv_drcp_rd",
    "drcp_rn",
    "__nv_drcp_rn",
    "drcp_ru",
    "__nv_drcp_ru",
    "drcp_rz",
    "__nv_drcp_rz",
    "dsqrt_rd",
    "__nv_dsqrt_rd",
    "dsqrt_rn",
    "__nv_dsqrt_rn",
    "dsqrt_ru",
    "__nv_dsqrt_ru",
    "dsqrt_rz",
    "__nv_dsqrt_rz",
    "erf",
    "__nv_erf",
    "erfc",
    "__nv_erfc",
    "erfcf",
    "__nv_erfcf",
    "erfcinv",
    "__nv_erfcinv",
    "erfcinvf",
    "__nv_erfcinvf",
    "erfcx",
    "__nv_erfcx",
    "erfcxf",
    "__nv_erfcxf",
    "erff",
    "__nv_erff",
    "erfinv",
    "__nv_erfinv",
    "erfinvf",
    "__nv_erfinvf",
    "exp",
    "__nv_exp",
    "exp10",
    "__nv_exp10",
    "exp10f",
    "__nv_exp10f",
    "exp2",
    "__nv_exp2",
    "exp2f",
    "__nv_exp2f",
    "expf",
    "__nv_expf",
    "expm1",
    "__nv_expm1",
    "expm1f",
    "__nv_expm1f",
    "fabs",
    "__nv_fabs",
    "fabsf",
    "__nv_fabsf",
    "fadd_rd",
    "__nv_fadd_rd",
    "fadd_rn",
    "__nv_fadd_rn",
    "fadd_ru",
    "__nv_fadd_ru",
    "fadd_rz",
    "__nv_fadd_rz",
    "fast_cosf",
    "__nv_fast_cosf",
    "fast_exp10f",
    "__nv_fast_exp10f",
    "fast_expf",
    "__nv_fast_expf",
    "fast_fdividef",
    "__nv_fast_fdividef",
    "fast_log10f",
    "__nv_fast_log10f",
    "fast_log2f",
    "__nv_fast_log2f",
    "fast_logf",
    "__nv_fast_logf",
    "fast_powf",
    "__nv_fast_powf",
    "__nv_fast_sincosf",
    "fast_sinf",
    "__nv_fast_sinf",
    "fast_tanf",
    "__nv_fast_tanf",
    "fdim",
    "__nv_fdim",
    "fdimf",
    "__nv_fdimf",
    "fdiv_rd",
    "__nv_fdiv_rd",
    "fdiv_rn",
    "__nv_fdiv_rn",
    "fdiv_ru",
    "__nv_fdiv_ru",
    "fdiv_rz",
    "__nv_fdiv_rz",
    "ffs",
    "__nv_ffs",
    "ffsll",
    "__nv_ffsll",
    "finitef",
    "__nv_finitef",
    "float2half_rn",
    "__nv_float2half_rn",
    "float2int_rd",
    "__nv_float2int_rd",
    "float2int_rn",
    "__nv_float2int_rn",
    "float2int_ru",
    "__nv_float2int_ru",
    "float2int_rz",
    "__nv_float2int_rz",
    "float2ll_rd",
    "__nv_float2ll_rd",
    "float2ll_rn",
    "__nv_float2ll_rn",
    "float2ll_ru",
    "__nv_float2ll_ru",
    "float2ll_rz",
    "__nv_float2ll_rz",
    "float2uint_rd",
    "__nv_float2uint_rd",
    "float2uint_rn",
    "__nv_float2uint_rn",
    "float2uint_ru",
    "__nv_float2uint_ru",
    "float2uint_rz",
    "__nv_float2uint_rz",
    "float2ull_rd",
    "__nv_float2ull_rd",
    "float2ull_rn",
    "__nv_float2ull_rn",
    "float2ull_ru",
    "__nv_float2ull_ru",
    "float2ull_rz",
    "__nv_float2ull_rz",
    "float_as_int",
    "__nv_float_as_int",
    "floor",
    "__nv_floor",
    "floorf",
    "__nv_floorf",
    "fma",
    "__nv_fma",
    "fma_rd",
    "__nv_fma_rd",
    "fma_rn",
    "__nv_fma_rn",
    "fma_ru",
    "__nv_fma_ru",
    "fma_rz",
    "__nv_fma_rz",
    "fmaf",
    "__nv_fmaf",
    "fmaf_rd",
    "__nv_fmaf_rd",
    "fmaf_rn",
    "__nv_fmaf_rn",
    "fmaf_ru",
    "__nv_fmaf_ru",
    "fmaf_rz",
    "__nv_fmaf_rz",
    "fmax",
    "__nv_fmax",
    "fmaxf",
    "__nv_fmaxf",
    "fmin",
    "__nv_fmin",
    "fminf",
    "__nv_fminf",
    "fmod",
    "__nv_fmod",
    "fmodf",
    "__nv_fmodf",
    "fmul_rd",
    "__nv_fmul_rd",
    "fmul_rn",
    "__nv_fmul_rn",
    "fmul_ru",
    "__nv_fmul_ru",
    "fmul_rz",
    "__nv_fmul_rz",
    "frcp_rd",
    "__nv_frcp_rd",
    "frcp_rn",
    "__nv_frcp_rn",
    "frcp_ru",
    "__nv_frcp_ru",
    "frcp_rz",
    "__nv_frcp_rz",
    "__nv_frexp",
    "__nv_frexpf",
    "frsqrt_rn",
    "__nv_frsqrt_rn",
    "fsqrt_rd",
    "__nv_fsqrt_rd",
    "fsqrt_rn",
    "__nv_fsqrt_rn",
    "fsqrt_ru",
    "__nv_fsqrt_ru",
    "fsqrt_rz",
    "__nv_fsqrt_rz",
    "fsub_rd",
    "__nv_fsub_rd",
    "fsub_rn",
    "__nv_fsub_rn",
    "fsub_ru",
    "__nv_fsub_ru",
    "fsub_rz",
    "__nv_fsub_rz",
    "hadd",
    "__nv_hadd",
    "half2float",
    "__nv_half2float",
    "hiloint2double",
    "__nv_hiloint2double",
    "hypot",
    "__nv_hypot",
    "hypotf",
    "__nv_hypotf",
    "ilogb",
    "__nv_ilogb",
    "ilogbf",
    "__nv_ilogbf",
    "int2double_rn",
    "__nv_int2double_rn",
    "int2float_rd",
    "__nv_int2float_rd",
    "int2float_rn",
    "__nv_int2float_rn",
    "int2float_ru",
    "__nv_int2float_ru",
    "int2float_rz",
    "__nv_int2float_rz",
    "int_as_float",
    "__nv_int_as_float",
    "isfinited",
    "__nv_isfinited",
    "isinfd",
    "__nv_isinfd",
    "isinff",
    "__nv_isinff",
    "isnand",
    "__nv_isnand",
    "isnanf",
    "__nv_isnanf",
    "j0",
    "__nv_j0",
    "j0f",
    "__nv_j0f",
    "j1",
    "__nv_j1",
    "j1f",
    "__nv_j1f",
    "jn",
    "__nv_jn",
    "jnf",
    "__nv_jnf",
    "ldexp",
    "__nv_ldexp",
    "ldexpf",
    "__nv_ldexpf",
    "lgamma",
    "__nv_lgamma",
    "lgammaf",
    "__nv_lgammaf",
    "ll2double_rd",
    "__nv_ll2double_rd",
    "ll2double_rn",
    "__nv_ll2double_rn",
    "ll2double_ru",
    "__nv_ll2double_ru",
    "ll2double_rz",
    "__nv_ll2double_rz",
    "ll2float_rd",
    "__nv_ll2float_rd",
    "ll2float_rn",
    "__nv_ll2float_rn",
    "ll2float_ru",
    "__nv_ll2float_ru",
    "ll2float_rz",
    "__nv_ll2float_rz",
    "llabs",
    "__nv_llabs",
    "llmax",
    "__nv_llmax",
    "llmin",
    "__nv_llmin",
    "llrint",
    "__nv_llrint",
    "llrintf",
    "__nv_llrintf",
    "llround",
    "__nv_llround",
    "llroundf",
    "__nv_llroundf",
    "log",
    "__nv_log",
    "log10",
    "__nv_log10",
    "log10f",
    "__nv_log10f",
    "log1p",
    "__nv_log1p",
    "log1pf",
    "__nv_log1pf",
    "log2",
    "__nv_log2",
    "log2f",
    "__nv_log2f",
    "logb",
    "__nv_logb",
    "logbf",
    "__nv_logbf",
    "logf",
    "__nv_logf",
    "longlong_as_double",
    "__nv_longlong_as_double",
    "max",
    "__nv_max",
    "min",
    "__nv_min",
    "__nv_modf",
    "__nv_modff",
    "mul24",
    "__nv_mul24",
    "mul64hi",
    "__nv_mul64hi",
    "mulhi",
    "__nv_mulhi",
    "nearbyint",
    "__nv_nearbyint",
    "nearbyintf",
    "__nv_nearbyintf",
    "nextafter",
    "__nv_nextafter",
    "nextafterf",
    "__nv_nextafterf",
    "normcdf",
    "__nv_normcdf",
    "normcdff",
    "__nv_normcdff",
    "normcdfinv",
    "__nv_normcdfinv",
    "normcdfinvf",
    "__nv_normcdfinvf",
    "popc",
    "__nv_popc",
    "popcll",
    "__nv_popcll",
    "pow",
    "__nv_pow",
    "powf",
    "__nv_powf",
    "powi",
    "__nv_powi",
    "powif",
    "__nv_powif",
    "rcbrt",
    "__nv_rcbrt",
    "rcbrtf",
    "__nv_rcbrtf",
    "remainder",
    "__nv_remainder",
    "remainderf",
    "__nv_remainderf",
    "__nv_remquo",
    "__nv_remquof",
    "rhadd",
    "__nv_rhadd",
    "rint",
    "__nv_rint",
    "rintf",
    "__nv_rintf",
    "round",
    "__nv_round",
    "roundf",
    "__nv_roundf",
    "rsqrt",
    "__nv_rsqrt",
    "rsqrtf",
    "__nv_rsqrtf",
    "sad",
    "__nv_sad",
    "saturatef",
    "__nv_saturatef",
    "scalbn",
    "__nv_scalbn",
    "scalbnf",
    "__nv_scalbnf",
    "signbitd",
    "__nv_signbitd",
    "signbitf",
    "__nv_signbitf",
    "sin",
    "__nv_sin",
    "__nv_sincos",
    "__nv_sincosf",
    "__nv_sincospi",
    "__nv_sincospif",
    "sinf",
    "__nv_sinf",
    "sinh",
    "__nv_sinh",
    "sinhf",
    "__nv_sinhf",
    "sinpi",
    "__nv_sinpi",
    "sinpif",
    "__nv_sinpif",
    "sqrt",
    "__nv_sqrt",
    "sqrtf",
    "__nv_sqrtf",
    "tan",
    "__nv_tan",
    "tanf",
    "__nv_tanf",
    "tanh",
    "__nv_tanh",
    "tanhf",
    "__nv_tanhf",
    "tgamma",
    "__nv_tgamma",
    "tgammaf",
    "__nv_tgammaf",
    "trunc",
    "__nv_trunc",
    "truncf",
    "__nv_truncf",
    "uhadd",
    "__nv_uhadd",
    "uint2double_rn",
    "__nv_uint2double_rn",
    "uint2float_rd",
    "__nv_uint2float_rd",
    "uint2float_rn",
    "__nv_uint2float_rn",
    "uint2float_ru",
    "__nv_uint2float_ru",
    "uint2float_rz",
    "__nv_uint2float_rz",
    "ull2double_rd",
    "__nv_ull2double_rd",
    "ull2double_rn",
    "__nv_ull2double_rn",
    "ull2double_ru",
    "__nv_ull2double_ru",
    "ull2double_rz",
    "__nv_ull2double_rz",
    "ull2float_rd",
    "__nv_ull2float_rd",
    "ull2float_rn",
    "__nv_ull2float_rn",
    "ull2float_ru",
    "__nv_ull2float_ru",
    "ull2float_rz",
    "__nv_ull2float_rz",
    "ullmax",
    "__nv_ullmax",
    "ullmin",
    "__nv_ullmin",
    "umax",
    "__nv_umax",
    "umin",
    "__nv_umin",
    "umul24",
    "__nv_umul24",
    "umul64hi",
    "__nv_umul64hi",
    "umulhi",
    "__nv_umulhi",
    "urhadd",
    "__nv_urhadd",
    "usad",
    "__nv_usad",
    "y0",
    "__nv_y0",
    "y0f",
    "__nv_y0f",
    "y1",
    "__nv_y1",
    "y1f",
    "__nv_y1f",
    "yn",
    "__nv_yn",
    "ynf",
    "__nv_ynf",
)
