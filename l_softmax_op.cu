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
#if GOOGLE_CUDA == 1
#define EIGEN_USE_GPU
#include "l_softmax_op.h"
#include "common.h"

#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"

using namespace tensorflow;

#include <cstdint>
#include <cmath>
#include <cfloat>

static __device__ int32_t cuda_factorial(int32_t n){
    int32_t frac = 1;
    while(n-- > 0) frac *= (n+1);
    return frac;
}

template <typename T>
__global__ void LargeMarginSoftmaxCudaKernel(CudaLaunchConfig config, const T * features, const T * weights, const int32_t * global_step, const int32_t * labels,
    const int32_t batch_size, const int32_t num_dimensions, const int32_t output_dimensions,
    const float base, const float gamma, const float power, const float lambda_min, const int32_t margin_order, const bool b_angular,
    float * feat_norm, float * weights_norm, float * cos_theta, float * theta_seg, float * output_lambda, T * losses) {

    *output_lambda = tf_max(base * std::pow(1.f + gamma * global_step[0], -power), lambda_min);//999.1242;//

    for(int32_t index = 0;index < batch_size;++index){
        T temp_sum{0};
        const T *feat_along = features + index * num_dimensions;
        for(int32_t dim_ind = 0;dim_ind < num_dimensions;++dim_ind){
            temp_sum += ldg(feat_along + dim_ind) * ldg(feat_along + dim_ind);
        }
        feat_norm[index] = std::pow(static_cast<float>(temp_sum), .5);
    }
    for(int32_t index = 0;index < output_dimensions;++index){
        T temp_sum{0};
        const T *weights_along = weights + index * num_dimensions;
        for(int32_t dim_ind = 0;dim_ind < num_dimensions;++dim_ind){
            temp_sum += ldg(weights_along + dim_ind) * ldg(weights_along + dim_ind);
        }
        weights_norm[index] = b_angular ? 1. : std::pow(static_cast<float>(temp_sum), .5);
    }
    for(int32_t index = 0;index < margin_order;++index){
        theta_seg[index] = std::cos(_PI * index / margin_order);
    }
    theta_seg[margin_order] = -1.;

    CUDA_1D_KERNEL_LOOP(worker_index, config.virtual_thread_count) {
        const int32_t output_row = worker_index / output_dimensions;
        const int32_t output_col = worker_index % output_dimensions;

        float feat_norm_value = feat_norm[output_row];
        float *p_cos_theta = cos_theta + output_row * output_dimensions;

        const T *feat_start = features + output_row * num_dimensions;
        // get cos_theta for features and all weights rows
        for(int32_t col_ind = 0;col_ind < output_dimensions;++col_ind){
            const T *weights_start = weights + col_ind * num_dimensions;
            T inner_dot{0};
            for(int32_t index = 0;index < num_dimensions;++index){
              inner_dot += ldg(feat_start + index) * ldg(weights_start + index);
            }
            p_cos_theta[col_ind] = static_cast<float>(inner_dot) / (feat_norm[output_row] * weights_norm[col_ind]);
        }
        int32_t k_block = 0;
        for(int32_t index = 1;index < margin_order+1;++index){
          if(p_cos_theta[output_col] > theta_seg[index]){
            k_block = index - 1;
            break;
          }
        }
        float single_cos = p_cos_theta[output_col];
        float cos_n_theta = std::pow(single_cos, margin_order*1.);
        float sin2_theta = 1. - single_cos * single_cos;
        for(int32_t m = 1;m <= margin_order/2; ++m){
          cos_n_theta += std::pow(-1, m) * std::pow(sin2_theta, m * 1.) * std::pow(single_cos, margin_order - 2.*m) * cuda_factorial(margin_order)/(cuda_factorial(2*m)*cuda_factorial(margin_order-2*m)*1.);
        }
        cos_n_theta = std::pow(-1., k_block) * cos_n_theta - 2 * k_block;
        if(ldg(labels + output_row) != output_col) cos_n_theta = 0.;
        *(losses + worker_index) = (feat_norm_value * weights_norm[output_col]) * (p_cos_theta[output_col] * output_lambda[0]/(output_lambda[0] + 1.) + cos_n_theta / (1. + output_lambda[0]));
    }
}

template <typename T>
void LargeMarginSoftmaxFunctor<GPUDevice, T>::operator()(OpKernelContext* context, const GPUDevice& d, typename TTypes<T>::ConstFlat features, typename TTypes<T>::ConstFlat weights, typename TTypes<int32_t>::ConstFlat global_step, typename TTypes<int32_t>::ConstFlat labels,
        const int32_t batch_size, const int32_t num_dimensions, const int32_t output_dimensions,
        const float base, const float gamma, const float power, const float lambda_min, const int32_t margin_order, const bool b_angular,
        typename TTypes<float>::Flat feat_norm, typename TTypes<float>::Flat weights_norm,
        typename TTypes<float>::Flat cos_theta, typename TTypes<float>::Flat theta_seg,
        typename TTypes<float>::Flat output_lambda, typename TTypes<T>::Flat losses) {

    CudaLaunchConfig config = GetCudaLaunchConfig(batch_size * output_dimensions, d);
    LargeMarginSoftmaxCudaKernel <<<config.block_count,
                        config.thread_per_block, 0, d.stream()>>> (config, features.data(), weights.data(), global_step.data(), labels.data(),
                            batch_size, num_dimensions, output_dimensions, base, gamma, power, lambda_min, margin_order, b_angular,
                            feat_norm.data(), weights_norm.data(), cos_theta.data(), theta_seg.data(), output_lambda.data(), losses.data());

    cudaError_t err = cudaGetLastError();
    if(cudaSuccess != err)
    {
      fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
      exit( -1 );
    }
}

template struct LargeMarginSoftmaxFunctor<GPUDevice, float>;
// #define DEFINE_GPU_SPECS(T)   \
//   template struct LargeMarginSoftmaxFunctorGPU<T>;

// TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#endif  // GOOGLE_CUDA
