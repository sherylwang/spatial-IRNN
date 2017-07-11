// ------------------------------------------------------------------
// SIAMESE RECURRENT ARCHITECTURE FOR VISUAL TRACKING
// Version 1.0, Copyright(c) July, 2017
// Xiaqing Xu, Bingpeng Ma, Hong Chang, Xilin Chen
// Written by Xiaqing Xu
// ------------------------------------------------------------------

#ifndef CAFFE_SPATIAL_IRNN_LAYER_HPP_
#define CAFFE_SPATIAL_IRNN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{

/**
*@brief Spatial-IRNN layers including IRNN layers of four different directions:
*up, down, left and right. 

*IRNNs are placed along each column for up and down IRNN layers, while they 
*are placed along each row for left and right layers. In this implementation,
*we actually merge all IRNNs into one in each layer for the convenience of 
*parallel computing. In each direction, the merged IRNN moves forward in 
*this direction during the forward propagation and moves backward in the 
*opposite direction during the backward propagation. 
*/
template <typename Dtype>
void Relu(const int count, Dtype* bottom_data);

template <typename Dtype>
void Grelu(const int count, Dtype* bottom_diff, const Dtype* top_data);

template <typename Dtype>
class RNNUPLayer : public Layer<Dtype>{
 public:
  explicit RNNUPLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RNNUP"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int N_;  
  int NH_; // output channels
  int NX_; // input channels
  int M_;  // height*width
  int H_;  // height
  int W_;  // width
  Blob<Dtype> cache_; // used during backpropagation, cache.data for f_diff, cache.diff for h_diff  
  Blob<Dtype> hh_; // used during backpropagation, hh_.diff for hidden state to hidden state's diff
 }; 
 
template <typename Dtype>
class RNNDOWNLayer : public Layer<Dtype>{
 public:
  explicit RNNDOWNLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RNNDOWN"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int N_;  
  int NH_; 
  int NX_; 
  int M_;  
  int H_;  
  int W_;  
  Blob<Dtype> cache_;  
  Blob<Dtype> hh_; 
 }; 

template <typename Dtype>
class RNNLEFTLayer : public Layer<Dtype>{
 public:
  explicit RNNLEFTLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RNNLEFT"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int N_;  
  int NH_;
  int NX_;
  int M_; 
  int H_; 
  int W_;  
  Blob<Dtype>  cache_;  
  Blob<Dtype>  hh_; 
 }; 
 

template <typename Dtype>
class RNNRIGHTLayer : public Layer<Dtype>{
 public:
  explicit RNNRIGHTLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RNNRIGHT"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 1; }
  
 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  int N_;  
  int NH_; 
  int NX_; 
  int M_;  
  int H_;  
  int W_;  
  Blob<Dtype>  cache_; 
  Blob<Dtype>  hh_; 
};

}  // namespace caffe

#endif // CAFFE_SPATIAL_IRNN_LAYER_HPP_
