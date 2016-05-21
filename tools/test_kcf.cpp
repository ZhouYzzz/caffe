#include "caffe/caffe.hpp"
#include "caffe/kcf/fft.hpp"
#include "cufft.h"

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = 1;
  LOG(INFO) << "Test KCF.";
  FFT f(1,20,20);
  return 0;
}
