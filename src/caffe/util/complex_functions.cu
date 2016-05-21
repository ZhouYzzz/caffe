#include "caffe/common.hpp"
#include "caffe/util/complex_functions.hpp"

namespace caffe {

__global__ void complex_kernel(const int n, const float* a, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = make_cuComplex(a[index], 0); }
}
__global__ void creal_kernel(const int n, const cuComplex* a, float* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = cuCrealf(a[index]); }
}
__global__ void cimag_kernel(const int n, const cuComplex* a, float* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = cuCimagf(a[index]); }
}
__global__ void cconj_kernel(const int n, const cuComplex* a, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = cuConjf(a[index]); }
}
__global__ void cadd_scalar_kernel(const int n, const cuComplex alpha, cuComplex* X) {
  CUDA_KERNEL_LOOP(index, n) { X[index] = cuCaddf(X[index], alpha); }
}
__global__ void cadd_kernel(const int n, const cuComplex* a, const cuComplex* b, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = cuCaddf(a[index], b[index]); }
}
__global__ void csub_kernel(const int n, const cuComplex* a, const cuComplex* b, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = cuCsubf(a[index], b[index]); }
}
__global__ void cmul_kernel(const int n, const cuComplex* a, const cuComplex* b, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = cuCmulf(a[index], b[index]); }
}
__global__ void cdiv_kernel(const int n, const cuComplex* a, const cuComplex* b, cuComplex* y) {
  CUDA_KERNEL_LOOP(index, n) { y[index] = cuCdivf(a[index], b[index]); }
}
void caffe_gpu_complex(const int N, const float* a, cuComplex* y) {
  complex_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
}
void caffe_gpu_creal(const int N, const cuComplex* a, float* y) {
  creal_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
}
void caffe_gpu_cimag(const int N, const cuComplex* a, float* y) {
  cimag_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
}
void caffe_gpu_cconj(const int N, const cuComplex* a, cuComplex* y) {
  cconj_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, y);
}
void caffe_gpu_cadd_scalar(const int N, const cuComplex alpha, cuComplex* X) {
  cadd_scalar_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, alpha, X);
}
void caffe_gpu_cadd(const int N, const cuComplex* a, const cuComplex* b, cuComplex* y) {
  cadd_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
}
void caffe_gpu_csub(const int N, const cuComplex* a, const cuComplex* b, cuComplex* y) {
  csub_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
}
void caffe_gpu_cmul(const int N, const cuComplex* a, const cuComplex* b, cuComplex* y) {
  cmul_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
}
void caffe_gpu_cdiv(const int N, const cuComplex* a, const cuComplex* b, cuComplex* y) {
  cdiv_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, b, y);
}

}
