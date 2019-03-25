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
#include "l_softmax_op.h"
#include "common.h"
#include "work_sharder.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>

using namespace tensorflow;

REGISTER_OP("LargeMarginSoftmaxGrad")
    .Attr("T: {float}")
    .Attr("margin_order: int")
    .Input("features: T")
    .Input("weights: T")
    .Input("labels: int32")
    .Input("back_grads: T")
    .Input("cur_lambda: float")
    .Output("grads_features: T")
    .Output("grads_weights: T")
    .Doc(R"doc(
        LargeMarginSoftmaxGrad is the Gradient op of LargeMarginSoftmax.
        The input features should has shape [N, D], where D is the dimension of the input features, N is the number of the input samples.
        The input weights should has shape [M, D], where D is the same as the second dimension of input features, while M is the outputs dimensions.
        The input labels should has shape [N].
        The input back_grads should in shape [N, M].
        The input cur_lambda should be one scalar.
        )doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return Status::OK();
    });

REGISTER_OP("AngularSoftmaxGrad")
    .Attr("T: {float}")
    .Attr("margin_order: int")
    .Input("features: T")
    .Input("weights: T")
    .Input("labels: int32")
    .Input("back_grads: T")
    .Input("cur_lambda: float")
    .Output("grads_features: T")
    .Output("grads_weights: T")
    .Doc(R"doc(
        AngularSoftmaxGrad is the Gradient op of AngularSoftmax.
        The input features should has shape [N, D], where D is the dimension of the input features, N is the number of the input samples.
        The input weights should has shape [M, D], where D is the same as the second dimension of input features, while M is the outputs dimensions.
        The input labels should has shape [N].
        The input back_grads should in shape [N, M].
        The input cur_lambda should be one scalar.
        )doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      c->set_output(1, c->input(1));
      return Status::OK();
    });

// CPU specialization of actual computation.
//template <typename T>
template <typename T>
struct LargeMarginSoftmaxGradFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const CPUDevice& d, typename TTypes<T>::ConstFlat back_grads, typename TTypes<T>::ConstFlat features, typename TTypes<T>::ConstFlat weights, typename TTypes<float>::ConstFlat cur_lambda, typename TTypes<int32_t>::ConstFlat labels,
        const int32_t batch_size, const int32_t num_dimensions, const int32_t output_dimensions, const int32_t margin_order, const bool b_angular,
        typename TTypes<float>::Flat feat_norm, typename TTypes<float>::Flat weights_norm,
        typename TTypes<float>::Flat cos_theta, typename TTypes<float>::Flat theta_seg,
        typename TTypes<T>::Flat grad_features, typename TTypes<T>::Flat grad_weights) {
    float *p_feat_norm = feat_norm.data();
    for(int32_t index = 0;index < batch_size;++index){
        T temp_sum{0};
        const T *feat_along = features.data() + index * num_dimensions;
        for(int32_t dim_ind = 0;dim_ind < num_dimensions;++dim_ind){
            temp_sum += feat_along[dim_ind] * feat_along[dim_ind];
        }
        p_feat_norm[index] = std::pow(static_cast<float>(temp_sum), .5);
    }
    float *p_weights_norm = weights_norm.data();
    for(int32_t index = 0;index < output_dimensions;++index){
        T temp_sum{0};
        const T *weights_along = weights.data() + index * num_dimensions;
        for(int32_t dim_ind = 0;dim_ind < num_dimensions;++dim_ind){
            temp_sum += weights_along[dim_ind] * weights_along[dim_ind];
        }
        p_weights_norm[index] = b_angular ? 1. : std::pow(static_cast<float>(temp_sum), .5);
    }
    float *p_theta_seg = theta_seg.data();
    for(int32_t index = 0;index < margin_order;++index){
        p_theta_seg[index] = std::cos(_PI * index / margin_order);
    }
    p_theta_seg[margin_order] = -1.;

    grad_features = grad_features.setZero();
    grad_weights = grad_weights.setZero();
    auto get_cosine_routine = [&features, &weights, &feat_norm, &weights_norm, &cos_theta, num_dimensions, output_dimensions](int64_t start, int64_t limit){
      for (int64_t worker_index = start; worker_index < limit; ++worker_index){
        const int32_t output_row = worker_index / output_dimensions;
        const int32_t output_col = worker_index % output_dimensions;

        const T *feat_start = features.data() + output_row * num_dimensions;
        const T *weights_start = weights.data() + output_col * num_dimensions;

        T inner_dot{0};
        for(int32_t index = 0;index < num_dimensions;++index){
          inner_dot += (feat_start[index] * weights_start[index]);
        }

        *(cos_theta.data() + worker_index) = static_cast<float>(inner_dot) / (feat_norm.data()[output_row] * weights_norm.data()[output_col]);
      }
    };

    auto get_loss_routine = [&back_grads, &features, &weights, &labels, &feat_norm, &weights_norm, &cos_theta, &theta_seg, &cur_lambda, &grad_features, &grad_weights, batch_size, num_dimensions, output_dimensions, margin_order, b_angular](int64_t start, int64_t limit){

      for (int64_t worker_index = start; worker_index < limit; ++worker_index){
        const int32_t output_row = worker_index / output_dimensions;
        const int32_t output_col = worker_index % output_dimensions;

        float feat_norm_value = feat_norm.data()[output_row];
        T *p_weights_norm = weights_norm.data();
        float *p_cos_theta = cos_theta.data() + output_row * output_dimensions;

        int32_t k_block = 0;
        for(int32_t index = 1;index < margin_order+1;++index){
          if(p_cos_theta[output_col] > theta_seg.data()[index]){
            k_block = index - 1;
            break;
          }
        }
        float single_cos = p_cos_theta[output_col];
        float sin2_theta = 1. - single_cos * single_cos;
        float cos_n_theta = 0.;
        // calculate cons_n_theta
        if(labels.data()[output_row] == output_col){
          cos_n_theta = std::pow(single_cos, margin_order*1.);
          for(int32_t m = 1; m <= margin_order / 2; ++m){
            float binomial = _factorial(margin_order) / (_factorial(2 * m) * _factorial(margin_order - 2 * m) * 1.);
            cos_n_theta += std::pow(-1, m) * std::pow(sin2_theta, m * 1.) * std::pow(single_cos, margin_order - 2. * m) * binomial;
          }
          cos_n_theta = std::pow(-1., k_block) * cos_n_theta - 2 * k_block;
        }
        // grad of cos_n_theta by cos_theta
        float grad_of_cos_theta = margin_order * std::pow(single_cos, margin_order - 1.);
        for(int32_t m = 1; m <= margin_order / 2; ++m){
          float binomial = _factorial(margin_order) / (_factorial(2 * m) * _factorial(margin_order - 2 * m) * 1.);
          grad_of_cos_theta += std::pow(-1, m) * std::pow(sin2_theta, m - 1.) * std::pow(single_cos, margin_order - 2 * m - 1.) * (-2 * m + margin_order - margin_order * std::pow(single_cos, 2.)) * binomial;
        }
        grad_of_cos_theta = grad_of_cos_theta * std::pow(-1., k_block);
        // backprop
        const float input_grad = *(back_grads.data() + worker_index);
        const T *feat_start = features.data() + output_row * num_dimensions;
        const T *weights_start = weights.data() + output_col * num_dimensions;

        T *grad_feat_start = grad_features.data() + output_row * num_dimensions;
        T *grad_weights_start = grad_weights.data() + output_col * num_dimensions;
        // softmax
        for(int32_t dim_ind = 0; dim_ind < num_dimensions; ++dim_ind){
          atomic_float_add(grad_weights_start + dim_ind, input_grad * cur_lambda.data()[0]/(cur_lambda.data()[0] + 1.) * feat_start[dim_ind]);
          atomic_float_add(grad_feat_start + dim_ind, input_grad * cur_lambda.data()[0]/(cur_lambda.data()[0] + 1.) * weights_start[dim_ind]);
        }
        // large margin softmax
        if(labels.data()[output_row] == output_col){
          for(int32_t dim_ind = 0; dim_ind < num_dimensions; ++dim_ind){
            float wx_norm = feat_norm_value * p_weights_norm[output_col];

            //
            float grad_cos_n_theta_by_w = b_angular ? grad_of_cos_theta * feat_start[dim_ind] / feat_norm_value : grad_of_cos_theta / (feat_norm_value * std::pow(p_weights_norm[output_col], 2.)) *
                                          ( (feat_start[dim_ind] * p_weights_norm[output_col]) -
                                            (wx_norm * single_cos * weights_start[dim_ind] / p_weights_norm[output_col])
                                          );
            if(b_angular){
              atomic_float_add(grad_weights_start + dim_ind, input_grad * feat_norm_value/(cur_lambda.data()[0] + 1.) *grad_cos_n_theta_by_w );
            }else{
              atomic_float_add(grad_weights_start + dim_ind, input_grad * feat_norm_value/(cur_lambda.data()[0] + 1.) * (
                                            cos_n_theta * weights_start[dim_ind] / p_weights_norm[output_col] +
                                            grad_cos_n_theta_by_w * p_weights_norm[output_col]       ) );
            }

            float grad_cos_n_theta_by_x = grad_of_cos_theta / (p_weights_norm[output_col] * std::pow(feat_norm_value, 2.)) *
                                          ( (weights_start[dim_ind] * feat_norm_value) -
                                            (wx_norm * single_cos * feat_start[dim_ind] / feat_norm_value)
                                          );

            atomic_float_add(grad_feat_start + dim_ind, input_grad * p_weights_norm[output_col]/(cur_lambda.data()[0] + 1.) * (
                                          cos_n_theta * feat_start[dim_ind] / feat_norm_value +
                                          grad_cos_n_theta_by_x * feat_norm_value       ) );
          }
        }
      }
    };

    const DeviceBase::CpuWorkerThreads& worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size * output_dimensions, num_dimensions * 2, get_cosine_routine);
    Shard(worker_threads.num_threads, worker_threads.workers, batch_size * output_dimensions, output_dimensions + margin_order, get_loss_routine);
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class LargeMarginSoftmaxGradOp : public OpKernel {
 public:
  explicit LargeMarginSoftmaxGradOp(OpKernelConstruction* context) : OpKernel(context) {
    b_angular = string(type_string()).rfind("Angular", 0) == 0;

    OP_REQUIRES_OK(context, context->GetAttr("margin_order", &m_margin_order));
    OP_REQUIRES(context, m_margin_order > 0, errors::InvalidArgument("Need Attr margin_order >= 1, got ", m_margin_order));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& features_in = context->input(0);
    const Tensor& weights_in = context->input(1);
    const Tensor& lables_in = context->input(2);
    const Tensor& grads_in = context->input(3);
    const Tensor& cur_lambda = context->input(4);

    OP_REQUIRES(context, features_in.shape().dims() == 2, errors::InvalidArgument("input features must have shape [N, D]."));
    OP_REQUIRES(context, weights_in.shape().dims() == 2, errors::InvalidArgument("input weights must have shape [M, D]."));
    OP_REQUIRES(context, features_in.dim_size(1) == weights_in.dim_size(1), errors::InvalidArgument("both input features and weights shoule have the same length in second dimension."));
    OP_REQUIRES(context, grads_in.dim_size(1) == weights_in.dim_size(0) && grads_in.dim_size(0) == features_in.dim_size(0), errors::InvalidArgument("input grads must have shape [N, M]."));
    OP_REQUIRES(context, lables_in.shape().dims() == 1 && lables_in.dim_size(0) == features_in.dim_size(0), errors::InvalidArgument("input lables must have shape [N]."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(cur_lambda.shape()), errors::InvalidArgument("the input cur_lambda should be one scalar."));

    const int32_t batch_size = features_in.dim_size(0);
    const int32_t num_dimensions = features_in.dim_size(1);
    const int32_t output_dimensions = weights_in.dim_size(0);

    Tensor* grad_features = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {batch_size, num_dimensions}, &grad_features));
    Tensor* grad_weights = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {output_dimensions, num_dimensions}, &grad_weights));
    Tensor feat_norm;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {batch_size}, &feat_norm));
    Tensor weights_norm;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {output_dimensions}, &weights_norm));
    Tensor cos_theta;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {batch_size, output_dimensions}, &cos_theta));
    Tensor theta_seg;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {m_margin_order + 1}, &theta_seg));

    LargeMarginSoftmaxGradFunctor<Device, T>()(context, context->eigen_device<Device>(),
                                        grads_in.template flat<T>(),
                                        features_in.template flat<T>(), weights_in.template flat<T>(),
                                        cur_lambda.template flat<float>(), lables_in.template flat<int32_t>(),
                                        batch_size, num_dimensions, output_dimensions, m_margin_order, b_angular,
                                        feat_norm.template flat<float>(), weights_norm.template flat<float>(),
                                        cos_theta.template flat<float>(), theta_seg.template flat<float>(),
                                        grad_features->template flat<T>(), grad_weights->template flat<T>());
  }

private:
  int32_t m_margin_order;
  float m_base;
  float m_gamma;
  float m_power;
  float m_lambda_min;
  bool b_angular;
  //PersistentTensor cos_theta_lookup;
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("LargeMarginSoftmaxGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      LargeMarginSoftmaxGradOp<CPUDevice, T>);
REGISTER_CPU(float);

#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("AngularSoftmaxGrad").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      LargeMarginSoftmaxGradOp<CPUDevice, T>);
REGISTER_CPU(float);

// TF_CALL_NUMBER_TYPES(REGISTER_CPU);
// #undef REGISTER_CPU

// Register the GPU kernels.
#if GOOGLE_CUDA == 1
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("LargeMarginSoftmaxGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      LargeMarginSoftmaxGradOp<GPUDevice, T>);
REGISTER_GPU(float);

#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("AngularSoftmaxGrad").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      LargeMarginSoftmaxGradOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
