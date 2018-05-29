# Large-Margin Softmax Loss, Angular Softmax Loss, Additive Margin Softmax, ArcFaceLoss And FocalLoss In Tensorflow

This repository contains core codes of the reimplementation of the following papers in TensorFlow:

- [Large-Margin Softmax Loss for Convolutional Neural Networks](https://arxiv.org/abs/1612.02295)
- [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)
- [Additive Margin Softmax for Face Verification](https://arxiv.org/abs/1801.05599) or [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://arxiv.org/abs/1801.09414)
- [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://arxiv.org/abs/1801.07698)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

If your goal is to reproduce the results in the original paper, please use the official codes:

- [Large Margin Softmax Loss in ICML 2016](https://github.com/wy1iu/LargeMargin_Softmax_Loss)
- [Angular Softmax Loss in CVPR 2017](https://github.com/wy1iu/sphereface)
- [Additive Margin Softmax](https://github.com/happynear/AMSoftmax)
- [ArcFace: Additive Angular Margin Loss](https://github.com/deepinsight/insightface)
- [Focal Loss in ICCV 2017](https://github.com/facebookresearch/Detectron)

## ##

For using these Ops on your own machine:

- copy the header file "cuda\_config.h" from "your\_python\_path/site-packages/external/local\_config\_cuda/cuda/cuda/cuda\_config.h" to "your\_python\_path/site-packages/tensorflow/include/tensorflow/stream\_executor/cuda/cuda\_config.h".

- run the following script:

```sh
mkdir build
cd build && cmake ..
make
```

- run "test\_op.py" and check the numeric errors to test your install
- follow the below codes snippet to integrate this Op into your own code:
	- For Large Margin Softmax Loss:

	```python
	op_module = tf.load_op_library(so_lib_path)
	large_margin_softmax = op_module.large_margin_softmax

	@ops.RegisterGradient("LargeMarginSoftmax")
	def _large_margin_softmax_grad(op, grad, _):
	  '''The gradients for `LargeMarginSoftmax`.
	  '''
	  inputs_features = op.inputs[0]
	  inputs_weights = op.inputs[1]
	  inputs_labels = op.inputs[2]
	  cur_lambda = op.outputs[1]
	  margin_order = op.get_attr('margin_order')

	  grads = op_module.large_margin_softmax_grad(inputs_features, inputs_weights, inputs_labels, grad, cur_lambda[0], margin_order)
	  return [grads[0], grads[1], None, None]

	var_weights = tf.Variable(initial_value, trainable=True, name='lsoftmax_weights')
	result = large_margin_softmax(features, var_weights, labels, global_step, 4, 1000., 0.000025, 35., 0.)
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=result[0]))
	```

	- For Angular Softmax Loss:

	```python
	op_module = tf.load_op_library(so_lib_path)
	angular_softmax = op_module.angular_softmax

	@ops.RegisterGradient("AngularSoftmax")
	def _angular_softmax_grad(op, grad, _):
	  '''The gradients for `AngularSoftmax`.
	  '''
	  inputs_features = op.inputs[0]
	  inputs_weights = op.inputs[1]
	  inputs_labels = op.inputs[2]
	  cur_lambda = op.outputs[1]
	  margin_order = op.get_attr('margin_order')

	  grads = op_module.angular_softmax_grad(inputs_features, inputs_weights, inputs_labels, grad, cur_lambda[0], margin_order)
	  return [grads[0], grads[1], None, None]

	var_weights = tf.Variable(initial_value, trainable=True, name='asoftmax_weights')
	normed_var_weights = tf.nn.l2_normalize(var_weights, 1, 1e-10, name='weights_normed')
	result = angular_softmax(features, normed_var_weights, labels, global_step, 4, 1000., 0.000025, 35., 0.)
	loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=result[0]))
	```
	- For others just refer to this [script](https://github.com/HiKapok/tf.extra_losses/blob/master/py_loss.py).

All the codes was tested under TensorFlow 1.6, Python 3.5, Ubuntu 16.04 with CUDA 8.0. The outputs of these Ops in C++ had been compared with the original caffe codes' outputs, and the bias could be ignored. The gradients of this Op had been checked using [tf.test.compute\_gradient\_error](https://www.tensorflow.org/api_docs/python/tf/test/compute_gradient_error) and [tf.test.compute\_gradient](https://www.tensorflow.org/api_docs/python/tf/test/compute_gradient). While the others are implemented following the official implementation in Python Ops.

If you encountered some linkage problem when generating or loading *.so, you are highly recommended to read this section in the [official tourial](https://www.tensorflow.org/extend/adding_an_op#compile_the_op_using_your_system_compiler_tensorflow_binary_installation) to make sure you were using the same C++ ABI version.

Any contributions to this repo is welcomed.

## ##
MIT License