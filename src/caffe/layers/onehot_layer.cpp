#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/onehot_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void OnehotLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  N_ = this->layer_param_.onehot_param().num_output();
  CHECK_GT(N_, 0) << "OnehotLayer num_output must be positive.";
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void OnehotLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  M_ = bottom[0]->count();
  vector<int> top_shape = bottom[0]->shape();
  top_shape.push_back(N_);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void OnehotLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  int index;
  for (int m = 0; m < M_; ++m) {
    index = static_cast<int>(bottom_data[m]);
    DCHECK_GE(index, 0);
    DCHECK_LT(index, N_);
    DCHECK_EQ(static_cast<Dtype>(index), bottom_data[m]) << "non-integer input";
    for (int n = 0; n < N_; ++n) {
      if (n == index) {
        *(top_data + m * N_ + n) = 1;
      } else {
        *(top_data + m * N_ + n) = 0;
      }
    }
  }
}

template <typename Dtype>
void OnehotLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  CHECK(!propagate_down[0]) << "Can't backpropagate to EmbedLayer input.";
}

#ifdef CPU_ONLY
STUB_GPU(OnehotLayer);
#endif

INSTANTIATE_CLASS(OnehotLayer);
REGISTER_LAYER_CLASS(Onehot);

}  // namespace caffe
