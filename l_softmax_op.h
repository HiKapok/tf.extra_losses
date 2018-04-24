// MIT License

// Copyright (c) 2018 Changan Wang

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#ifndef KERNEL_L_SOFTMAX_H_
#define KERNEL_L_SOFTMAX_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/op_kernel.h"

#include <cstdint>
#include <tuple>
#include <limits>
#include <iostream>

using tensorflow::TTypes;
using tensorflow::OpKernelContext;

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#define _PI 3.14159265358979323846

template <typename Device, typename T>
struct LargeMarginSoftmaxFunctor {
  void operator()(OpKernelContext* context, const Device& d, typename TTypes<T>::ConstFlat features, typename TTypes<T>::ConstFlat weights, typename TTypes<int32_t>::ConstFlat global_step, typename TTypes<int32_t>::ConstFlat labels,
        const int32_t batch_size, const int32_t num_dimensions, const int32_t output_dimensions,
        const float base, const float gamma, const float power, const float lambda_min, const int32_t margin_order,
        typename TTypes<float>::Flat feat_norm, typename TTypes<float>::Flat weights_norm,
        typename TTypes<float>::Flat cos_theta, typename TTypes<float>::Flat theta_seg,
        typename TTypes<float>::Flat output_lambda, typename TTypes<T>::Flat losses);
};

template <typename Device, typename T>
struct LargeMarginSoftmaxGradFunctor {
  void operator()(OpKernelContext* context, const Device& d, typename TTypes<T>::ConstFlat back_grads, typename TTypes<T>::ConstFlat features, typename TTypes<T>::ConstFlat weights, typename TTypes<float>::ConstFlat cur_lambda, typename TTypes<int32_t>::ConstFlat labels,
        const int32_t batch_size, const int32_t num_dimensions, const int32_t output_dimensions, const int32_t margin_order,
        typename TTypes<float>::Flat feat_norm, typename TTypes<float>::Flat weights_norm,
        typename TTypes<float>::Flat cos_theta, typename TTypes<float>::Flat theta_seg,
        typename TTypes<T>::Flat grad_features, typename TTypes<T>::Flat grad_weights);
};

#if GOOGLE_CUDA == 1
template <typename T>
struct LargeMarginSoftmaxFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const GPUDevice& d, typename TTypes<T>::ConstFlat features, typename TTypes<T>::ConstFlat weights, typename TTypes<int32_t>::ConstFlat global_step, typename TTypes<int32_t>::ConstFlat labels,
        const int32_t batch_size, const int32_t num_dimensions, const int32_t output_dimensions,
        const float base, const float gamma, const float power, const float lambda_min, const int32_t margin_order,
        typename TTypes<float>::Flat feat_norm, typename TTypes<float>::Flat weights_norm,
        typename TTypes<float>::Flat cos_theta, typename TTypes<float>::Flat theta_seg,
        typename TTypes<float>::Flat output_lambda, typename TTypes<T>::Flat losses);
};
#endif

#if GOOGLE_CUDA == 1
template <typename T>
struct LargeMarginSoftmaxGradFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const GPUDevice& d, typename TTypes<T>::ConstFlat back_grads, typename TTypes<T>::ConstFlat features, typename TTypes<T>::ConstFlat weights, typename TTypes<float>::ConstFlat cur_lambda, typename TTypes<int32_t>::ConstFlat labels,
        const int32_t batch_size, const int32_t num_dimensions, const int32_t output_dimensions, const int32_t margin_order,
        typename TTypes<float>::Flat feat_norm, typename TTypes<float>::Flat weights_norm,
        typename TTypes<float>::Flat cos_theta, typename TTypes<float>::Flat theta_seg,
        typename TTypes<T>::Flat grad_features, typename TTypes<T>::Flat grad_weights);
};
#endif

#endif // KERNEL_L_SOFTMAX_H_

