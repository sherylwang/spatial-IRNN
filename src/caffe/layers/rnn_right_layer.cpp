// ------------------------------------------------------------------
// SIAMESE RECURRENT ARCHITECTURE FOR VISUAL TRACKING
// Version 1.0, Copyright(c) July, 2017
// Xiaqing Xu, Bingpeng Ma, Hong Chang, Xilin Chen
// Written by Xiaqing Xu
// ------------------------------------------------------------------

#include <iostream>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/spatial_irnn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
inline void Relu(const int count, Dtype *bottom_data) {
  for (int i = 0; i < count; ++i) {
    bottom_data[i] = std::max(bottom_data[i], Dtype(0.));
  }
}
template <typename Dtype>
inline void Grelu(const int count, Dtype *bottom_diff, const Dtype *top_data) {
  for (int i = 0; i < count; ++i) {
    bottom_diff[i] = Dtype(1.) * (top_data[i] > 0);
  }
}

template <typename Dtype>
void RNNRIGHTLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  W_ = bottom[0]->num();
  NX_ = bottom[0]->channels();
  NH_ = NX_;
  H_ = bottom[0]->height();
  N_ = bottom[0]->width();
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    vector<int> w_shape(2);
    w_shape[0] = NH_;
    w_shape[1] = NH_;
    this->blobs_[0].reset(new Blob<Dtype>(w_shape));
    shared_ptr<Filler<Dtype>> weight_filler(
        GetFiller<Dtype>(this->layer_param_.rnn_right_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
  }
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void RNNRIGHTLayer<Dtype>::Reshape(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  vector<int> top_shape = bottom[0]->shape();
  cache_.Reshape(top_shape);

  vector<int> hh_shape(2);
  hh_shape[0] = NH_;
  hh_shape[1] = H_ * N_;

  hh_.Reshape(hh_shape);
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void RNNRIGHTLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
    const vector<Blob<Dtype> *> &top) {
  const Dtype *bottom_data = bottom[0]->cpu_data(); 
  const int count = top[0]->count();

  const Dtype *w = this->blobs_[0]->cpu_data();
  Dtype *top_data = top[0]->mutable_cpu_data();

  caffe_copy(count, bottom_data, top_data);

  for (int i = 0; i < W_; i++) {
    if (i > 0) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NH_, H_ * N_, NH_,
          Dtype(1.), w, top_data + (i - 1) * NH_ * H_ * N_, Dtype(1.), 
          top_data + i * NH_ * H_ * N_);
    }
    Relu(NH_ * H_ * N_, top_data + i * NH_ * H_ * N_);
  }
}

template <typename Dtype>
void RNNRIGHTLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
    const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom) {
  const Dtype *top_diff = top[0]->cpu_diff();
  const Dtype *top_data = top[0]->cpu_data();
  const int count = bottom[0]->count();
  const Dtype *w = this->blobs_[0]->cpu_data();

  Dtype *w_diff = this->blobs_[0]->mutable_cpu_diff();
  // dh
  Dtype *h_diff = cache_.mutable_cpu_data();
  // f'(h)
  Dtype *f_diff = cache_.mutable_cpu_diff();

  Dtype *hh_diff = hh_.mutable_cpu_diff();

  Grelu(count, f_diff, top_data);

  caffe_copy(count, top_diff, h_diff);

  for (int i = W_ - 1; i >= 0; i--) {
    // dzdf
    caffe_mul(NH_ * H_ * N_, h_diff + i * NH_ * H_ * N_,
        f_diff + i * NH_ * H_ * N_, f_diff + i * NH_ * H_ * N_);
    // dzdhh
    caffe_cpu_gemm(CblasTrans, CblasNoTrans, NH_, H_ * N_, NH_, Dtype(1.), w,
        f_diff + i * NH_ * H_ * N_, Dtype(0.), hh_diff);

    if (i > 0) {
      caffe_add(NH_ * H_ * N_, hh_diff, h_diff + (i - 1) * NH_ * H_ * N_,
          h_diff + (i - 1) * NH_ * H_ * N_);
      caffe_cpu_gemm(CblasNoTrans, CblasTrans, NH_, NH_, H_ * N_, Dtype(1.),
          f_diff + i * NH_ * H_ * N_, top_data + (i - 1) * NH_ * H_ * N_,
          Dtype(1.), w_diff);
    }
  }

  if (propagate_down[0]) {
    Dtype *bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_copy(bottom[0]->count(), f_diff, bottom_diff);
  }
}

INSTANTIATE_CLASS(RNNRIGHTLayer);
REGISTER_LAYER_CLASS(RNNRIGHT);

} // namespace caffe
