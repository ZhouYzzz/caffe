#include <algorithm>
#include <vector>

#include "caffe/layers/yolo_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void YoloLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    LOG(INFO) << "[YOLO]LayerSetup";
    // set up your params
    side_ = layer_param_.yolo_loss_param().side();
    num_ = layer_param_.yolo_loss_param().num();
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    LOG(INFO) << "[YOLO]Reshape";
    // check shape and reshape blobs
    CHECK_EQ(bottom[0]->shape(), side_*num_);
    top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    LOG(INFO) << "[YOLO]Forward";
    const Dtype* data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* loss = top[0]->mutable_cpu_data();

    // calculate your loss here
    
    loss[0] = 0; // assign
}

template <typename Dtype>
void YoloLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    LOG(INFO) << "[YOLO]Backward";
    const Dtype* data = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* grad = bottom[0]->mutable_cpu_diff();

    // calculate your gradient here

    grad[0] = 0; // assign
}


#ifdef CPU_ONLY
STUB_GPU(YoloLossLayer);
#endif

INSTANTIATE_CLASS(YoloLossLayer);

}  // namespace caffe
