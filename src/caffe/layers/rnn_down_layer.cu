// ------------------------------------------------------------------
// SIAMESE RECURRENT ARCHITECTURE FOR VISUAL TRACKING
// Version 1.0, Copyright(c) July, 2017
// Xiaqing Xu, Bingpeng Ma, Hong Chang, Xilin Chen
// Written by Xiaqing Xu
// ------------------------------------------------------------------

#include <vector>
#include <iostream>

#include "caffe/filler.hpp"
#include "caffe/layers/spatial_irnn_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe{
template <typename Dtype>
__global__ void ReLUForward(const int n, Dtype* in) {
  CUDA_KERNEL_LOOP(index, n) {
    in[index] = in[index] > 0 ? in[index] : Dtype(0.);
  }
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, Dtype* out_diff,const Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = Dtype(1.) * (top_data[index] > 0);
  }
}

template <typename Dtype>
void RNNDOWNLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top){
  const Dtype* bottom_data = bottom[0]->gpu_data();
  const int count = top[0]->count();
  const Dtype* w = this->blobs_[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data(); 

  caffe_copy(count, bottom_data, top_data);
 
  for(int i = 0; i < H_; i++){
    if(i > 0){
      caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, NH_, W_ * N_, NH_, Dtype(1.),
          w, top_data + (i - 1) * NH_ * N_* W_ , Dtype(1.),
          top_data + i * NH_ * N_* W_);
  }
  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(NH_*W_*N_), CAFFE_CUDA_NUM_THREADS>>>(
      NH_ * W_ * N_, top_data + i * NH_ * N_* W_);
  CUDA_POST_KERNEL_CHECK;
  }
}

template <typename Dtype>
void RNNDOWNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){  
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const int count = bottom[0]->count();
  const Dtype* w = this->blobs_[0]->gpu_data();

  Dtype* w_diff = this->blobs_[0]->mutable_gpu_diff();
  // dh
  Dtype* h_diff = cache_.mutable_gpu_data();
  // f'(h)
  Dtype* f_diff = cache_.mutable_gpu_diff();

  Dtype* hh_diff = hh_.mutable_gpu_diff();

  ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, f_diff, top_data);
  CUDA_POST_KERNEL_CHECK;
  
  caffe_copy(count, top_diff, h_diff);

  for(int i = H_ - 1; i >= 0; i--){
    // dzdf
    caffe_gpu_mul(NH_ * W_ * N_, h_diff + i * NH_ * N_ * W_,
    f_diff + i * NH_ * N_ * W_, f_diff + i * NH_ * N_* W_);
    // dzdhh
    caffe_gpu_gemm(CblasTrans, CblasNoTrans, NH_, W_ * N_, NH_, Dtype(1.),
        w, f_diff + i * NH_ * N_* W_ , Dtype(0.), hh_diff);
    if(i > 0){
      caffe_gpu_add(NH_ * W_ * N_, hh_diff,
          h_diff + (i - 1)* NH_ * N_ * W_,
          h_diff + (i - 1)* NH_ * N_ * W_);
      caffe_gpu_gemm(CblasNoTrans, CblasTrans, NH_, NH_, W_ * N_, Dtype(1.),
          f_diff + i * NH_ * N_ * W_ , top_data + (i - 1) * NH_ * N_ * W_,
          Dtype(1.),  w_diff);
    }
  } 

  if(propagate_down[0]){
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_copy(bottom[0]->count(), f_diff, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(RNNDOWNLayer);
}  // namespace caffe
 


