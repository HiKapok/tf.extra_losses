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
#include "work_sharder.h"
#include "common.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

#include <cmath>
#include <limits>

using namespace tensorflow;

REGISTER_OP("LargeMarginSoftmax")
    .Attr("T: {float}")
    .Attr("margin_order: int")
    .Attr("base: float")
    .Attr("gamma: float")
    .Attr("power: float")
    .Attr("lambda_min: float")
    .Input("features: T")
    .Input("weights: T")
    .Input("labels: int32")
    .Input("global_step: int32")
    .Output("loss: T")
    .Output("cur_lambda: float")
    .Doc(R"doc(
        large_margin_softmax is a generalized large-margin softmax (L-Softmax) loss which explicitly encourages intra-class compactness and inter-class separability between learned features.
        The input features should has shape [N, D], where D is the dimension of the input features, N is the number of the input samples.
        The input weights should has shape [M, D], where D is the same as the second dimension of input features, while M is the outputs dimensions.
        The input labels should has shape [N].
        The base, gamma, power and lambda_min are parameters of exponential lambda descent: cur_lambda = base * pow(1. + gamma * global_step, -power).
        )doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle features_shape = c->input(0);
      shape_inference::DimensionHandle num_per_batch = c->Dim(features_shape, 0);
      shape_inference::DimensionHandle num_dimensions = c->Dim(features_shape, 1);
      shape_inference::DimensionHandle output_dimensions = c->Dim(c->input(1), 0);
      c->set_output(0, c->MakeShape({num_per_batch, output_dimensions}));
      // scalar output for lambda
      c->set_output(1, c->MakeShape({1}));
      return Status::OK();
    });

REGISTER_OP("AngularSoftmax")
    .Attr("T: {float}")
    .Attr("margin_order: int")
    .Attr("base: float")
    .Attr("gamma: float")
    .Attr("power: float")
    .Attr("lambda_min: float")
    .Input("features: T")
    .Input("weights: T")
    .Input("labels: int32")
    .Input("global_step: int32")
    .Output("loss: T")
    .Output("cur_lambda: float")
    .Doc(R"doc(
        angular_softmax is an improved version of large-margin softmax (L-Softmax) loss which explicitly encourages intra-class compactness and inter-class separability between learned features.
        The input features should has shape [N, D], where D is the dimension of the input features, N is the number of the input samples.
        The input weights should has shape [M, D], where D is the same as the second dimension of input features, while M is the outputs dimensions.
        The input labels should has shape [N].
        The base, gamma, power and lambda_min are parameters of exponential lambda descent: cur_lambda = base * pow(1. + gamma * global_step, -power).
        )doc")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      shape_inference::ShapeHandle features_shape = c->input(0);
      shape_inference::DimensionHandle num_per_batch = c->Dim(features_shape, 0);
      shape_inference::DimensionHandle num_dimensions = c->Dim(features_shape, 1);
      shape_inference::DimensionHandle output_dimensions = c->Dim(c->input(1), 0);
      c->set_output(0, c->MakeShape({num_per_batch, output_dimensions}));
      // scalar output for lambda
      c->set_output(1, c->MakeShape({1}));
      return Status::OK();
    });

// CPU specialization of actual computation.
//template <typename T>
template <typename T>
struct LargeMarginSoftmaxFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const CPUDevice& d, typename TTypes<T>::ConstFlat features, typename TTypes<T>::ConstFlat weights, typename TTypes<int32_t>::ConstFlat global_step, typename TTypes<int32_t>::ConstFlat labels,
                  const int32_t batch_size, const int32_t num_dimensions, const int32_t output_dimensions,
                  const float base, const float gamma, const float power, const float lambda_min, const int32_t margin_order, const bool b_angular,
                  typename TTypes<float>::Flat feat_norm, typename TTypes<float>::Flat weights_norm,
                  typename TTypes<float>::Flat cos_theta, typename TTypes<float>::Flat theta_seg,
                  typename TTypes<float>::Flat output_lambda, typename TTypes<T>::Flat losses) {

    *output_lambda.data() = std::max(base * std::pow(1.f + gamma * global_step.data()[0], -power), lambda_min);//999.1242;//

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
        //std::cout << *(cos_theta.data() + worker_index) << " " << std::endl;
      }
    };

    auto get_loss_routine = [&features, &weights, &labels, &feat_norm, &weights_norm, &cos_theta, &theta_seg, &output_lambda, &losses, batch_size, num_dimensions, output_dimensions, margin_order](int64_t start, int64_t limit){

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
        float cos_n_theta = std::pow(single_cos, margin_order*1.);

        float sin2_theta = 1. - single_cos * single_cos;
        for(int32_t m = 1;m <= margin_order/2;++m){
          cos_n_theta += std::pow(-1, m) * std::pow(sin2_theta, m * 1.) * std::pow(single_cos, margin_order - 2.*m) * _factorial(margin_order)/(_factorial(2*m)*_factorial(margin_order-2*m)*1.);
        }
        cos_n_theta = std::pow(-1., k_block) * cos_n_theta - 2 * k_block;
        if(labels.data()[output_row] != output_col) cos_n_theta = 0.;
        *(losses.data() + worker_index) = (feat_norm_value * p_weights_norm[output_col]) * (p_cos_theta[output_col] * output_lambda.data()[0]/(output_lambda.data()[0] + 1.) + cos_n_theta / (1. + output_lambda.data()[0]));
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
class LargeMarginSoftmaxOp : public OpKernel {
 public:
  explicit LargeMarginSoftmaxOp(OpKernelConstruction* context) : OpKernel(context) {
    b_angular = string(type_string()).rfind("Angular", 0) == 0;

    OP_REQUIRES_OK(context, context->GetAttr("margin_order", &m_margin_order));
    OP_REQUIRES(context, m_margin_order > 0, errors::InvalidArgument("Need Attr margin_order >= 1, got ", m_margin_order));

    OP_REQUIRES_OK(context, context->GetAttr("base", &m_base));
    OP_REQUIRES(context, m_base > 0., errors::InvalidArgument("Need Attr base > 0.0, got ", m_base));

    OP_REQUIRES_OK(context, context->GetAttr("gamma", &m_gamma));
    OP_REQUIRES(context, m_gamma > 0., errors::InvalidArgument("Need Attr gamma > 0.0, got ", m_gamma));

    OP_REQUIRES_OK(context, context->GetAttr("power", &m_power));
    OP_REQUIRES(context, m_power > 0., errors::InvalidArgument("Need Attr power > 0.0, got ", m_power));

    OP_REQUIRES_OK(context, context->GetAttr("lambda_min", &m_lambda_min));
    OP_REQUIRES(context, m_lambda_min >= 0., errors::InvalidArgument("Need Attr lambda_min >= 0.0, got ", m_lambda_min));

    //OP_REQUIRES_OK(context, context->allocate_persistent(DT_FLOAT, {m_margin_order + 1}, &cos_theta_lookup, nullptr));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& features_in = context->input(0);
    const Tensor& weights_in = context->input(1);
    const Tensor& lables_in = context->input(2);
    const Tensor& global_step_in = context->input(3);

    OP_REQUIRES(context, features_in.shape().dims() == 2, errors::InvalidArgument("input features must have shape [N, D]."));
    OP_REQUIRES(context, weights_in.shape().dims() == 2, errors::InvalidArgument("input weights must have shape [M, D]."));
    OP_REQUIRES(context, features_in.dim_size(1) == weights_in.dim_size(1), errors::InvalidArgument("both input features and weights shoule have the same length in second dimension."));
    OP_REQUIRES(context, lables_in.shape().dims() == 1 && lables_in.dim_size(0) == features_in.dim_size(0), errors::InvalidArgument("input lables must have shape [N]."));
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(global_step_in.shape()), errors::InvalidArgument("the input global_step should be one scalar."));

    const int32_t batch_size = features_in.dim_size(0);
    const int32_t num_dimensions = features_in.dim_size(1);
    const int32_t output_dimensions = weights_in.dim_size(0);

    Tensor* losses = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, {batch_size, output_dimensions}, &losses));
    Tensor* output_lambda = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, {1}, &output_lambda));
    Tensor feat_norm;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {batch_size}, &feat_norm));
    Tensor weights_norm;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {output_dimensions}, &weights_norm));
    Tensor cos_theta;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {batch_size, output_dimensions}, &cos_theta));
    Tensor theta_seg;
    OP_REQUIRES_OK(context, context->allocate_temp(DT_FLOAT, {m_margin_order + 1}, &theta_seg));

    LargeMarginSoftmaxFunctor<Device, T>()(context, context->eigen_device<Device>(),
                                        features_in.template flat<T>(), weights_in.template flat<T>(),
                                        global_step_in.template flat<int32_t>(), lables_in.template flat<int32_t>(),
                                        batch_size, num_dimensions, output_dimensions, m_base, m_gamma, m_power, m_lambda_min, m_margin_order, b_angular,
                                        feat_norm.template flat<float>(), weights_norm.template flat<float>(),
                                        cos_theta.template flat<float>(), theta_seg.template flat<float>(),
                                        output_lambda->template flat<float>(), losses->template flat<T>());
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
      Name("LargeMarginSoftmax").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      LargeMarginSoftmaxOp<CPUDevice, T>);
REGISTER_CPU(float);

#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("AngularSoftmax").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      LargeMarginSoftmaxOp<CPUDevice, T>);
REGISTER_CPU(float);

// TF_CALL_NUMBER_TYPES(REGISTER_CPU);
// #undef REGISTER_CPU

// Register the GPU kernels.
#if GOOGLE_CUDA == 1
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("LargeMarginSoftmax").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      LargeMarginSoftmaxOp<GPUDevice, T>);
REGISTER_GPU(float);

#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("AngularSoftmax").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      LargeMarginSoftmaxOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
