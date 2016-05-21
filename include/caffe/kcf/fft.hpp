#ifndef FFT_HPP_
#define FFT_HPP_

#include "caffe/common.hpp"
#include "glog/logging.h"

#include "cuComplex.h"
#include "cufft.h"

#define CUFFT_CHECK(condition) do{CHECK_EQ(condition,CUFFT_SUCCESS);}while(0)

class FFT {
public:
  FFT(int batch, int H, int W) {
    if (batch == 1)
      CUFFT_CHECK(cufftPlan2d(&plan, H, W, CUFFT_C2C));
    else {
      int R[2] = {H, W}; int N = H*W;
      CUFFT_CHECK(cufftPlanMany(&plan,2,R,NULL,1,N,NULL,1,N,CUFFT_C2C,batch));
    }
	LOG(INFO) <<"FFT init, size of "<<batch<<" * "<<H<<" * "<<W<<" .";
  }
  ~FFT() {
    CUFFT_CHECK(cufftDestroy(plan));
  }

  void fft2(cuComplex* a) {
    CUFFT_CHECK(cufftExecC2C(plan, a, a, CUFFT_FORWARD));
    cudaDeviceSynchronize();
  }
  void fft2(cuComplex* a, cuComplex* y) {
    CUFFT_CHECK(cufftExecC2C(plan, a, y, CUFFT_FORWARD));
    cudaDeviceSynchronize();
  }
  void ifft2(cuComplex* a) {
    CUFFT_CHECK(cufftExecC2C(plan, a, a, CUFFT_INVERSE));
    cudaDeviceSynchronize();
  }
  void ifft2(cuComplex* a, cuComplex* y) {
    CUFFT_CHECK(cufftExecC2C(plan, a, y, CUFFT_INVERSE));
    cudaDeviceSynchronize();
  }

protected:
  cufftHandle plan;
};

#endif
