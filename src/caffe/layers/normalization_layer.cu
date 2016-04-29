#include "normalization_layer.h"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NormalizationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
}

}