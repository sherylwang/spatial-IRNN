#ifndef CAFFE_STRA_RNN_LAYER_HPP_
#define CAFFE_STRA_RNN_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe{
/**
 * @brief This spatial RNN that implemented in the paper 
  "Inside-Outside Net: Detecting objects in Context with Skip
  Pooling and Recurrent Networks" 
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
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  #ifndef CPU_ONLY
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  #endif
  int N_;  //number of 
  int NH_; //output channel, equal to output
  int NX_; //input channels
  int M_;  //height*width
  int H_;  //height
  int W_;  //width

  Blob<Dtype>  cache_; //used for backpropgation, cache.data for f_diff, cache.diff for h_diff  
  Blob<Dtype> hh_; //hidden to hidden diff
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
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  #ifndef CPU_ONLY
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  #endif
  int N_;  //number of 
  int NH_; //output channel, equal to output
  int NX_; //input channels
  int M_;  //height*width
  int H_;  //height
  int W_;  //width

  Blob<Dtype> cache_; //used for backpropgation, cache.data for f_diff, cache.diff for h_diff  
  Blob<Dtype> hh_; //hidden to hidden diff
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
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  #ifndef CPU_ONLY
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  #endif
  int N_;  //number of 
  int NH_; //output channel, equal to output
  int NX_; //input channels
  int M_;  //height*width
  int H_;  //height
  int W_;  //width

  Blob<Dtype>  cache_; //used for backpropgation, cache.data for f_diff, cache.diff for h_diff  
  Blob<Dtype>  hh_; //hidden to hidden diff
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
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  #ifndef CPU_ONLY
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  #endif
  int N_;  //number of 
  int NH_; //output channel, equal to output
  int NX_; //input channels
  int M_;  //height*width
  int H_;  //height
  int W_;  //width

  Blob<Dtype>  cache_; //used for backpropgation, cache.data for f_diff, cache.diff for h_diff  
  Blob<Dtype>  hh_; //hidden to hidden diff
 }; 
 
 
 // namespace caffe
}
#endif // CAFFE_STRA_RNN_LAYER_HPP_
