#ifndef CAFFE_UTIL_COMPLEX_FUNCTIONS_H_
#define CAFFE_UTIL_COMPLEX_FUNCTIONS_H_

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

#include "cuComplex.h"

namespace caffe {

void caffe_gpu_complex(const int N, const float* a, cuComplex* y);

void caffe_gpu_creal(const int N, const cuComplex* a, float* y);

void caffe_gpu_cimag(const int N, const cuComplex* a, float* y);

void caffe_gpu_cconj(const int N, const cuComplex* a, cuComplex* y);

void caffe_gpu_cadd_scalar(const int N, const cuComplex alpha, cuComplex* X);

void caffe_gpu_cadd(const int N, const cuComplex* a, const cuComplex* b, cuComplex* y);

void caffe_gpu_csub(const int N, const cuComplex* a, const cuComplex* b, cuComplex* y);

void caffe_gpu_cmul(const int N, const cuComplex* a, const cuComplex* b, cuComplex* y);

void caffe_gpu_cdiv(const int N, const cuComplex* a, const cuComplex* b, cuComplex* y);

}

#endif  // CAFFE_UTIL_COMPLEX_FUNCTIONS_H_
