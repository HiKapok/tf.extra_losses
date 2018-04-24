# MIT License

# Copyright (c) 2018 Changan Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import shutil
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import math

LIB_NAME = 'l_softmax'

def load_op_module(lib_name):
  lib_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'build/lib{0}.so'.format(lib_name))
  oplib = tf.load_op_library(lib_path)
  return oplib

op_module = load_op_module(LIB_NAME)

features = [[0.1, 0.2, -0.3, -0.4], [-1.1, -1.2, 1.3, 1.4], [2.1, 2.2, -2.3, -2.4]]
labels = [3, 2, 1]
weights = [[0., 0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8, 0.9, 1.0, 1.1], [1.2, 1.3, 1.4, 1.5], [1.6, 1.7, 1.8, 1.9]]

class LargeMarginSoftmaxTest(tf.test.TestCase):
  def testLargeMarginSoftmax(self):
    with tf.device('/gpu:0'):
      # map C++ operators to python objects
      large_margin_softmax = op_module.large_margin_softmax
      result = large_margin_softmax(features, weights, labels, 1, 4, 1000., 0.000025, 35., 0.)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=result[0], name=None))
      with self.test_session() as sess:
        print('large_margin_softmax in gpu:', sess.run([loss, result[1]]))
    with tf.device('/cpu:0'):
      # map C++ operators to python objects
      large_margin_softmax = op_module.large_margin_softmax
      result = large_margin_softmax(features, weights, labels, 1, 4, 1000., 0.000025, 35., 0.)
      loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=result[0], name=None))
      with self.test_session() as sess:
        print('large_margin_softmax in cpu:', sess.run([loss, result[1]]))

@ops.RegisterGradient("LargeMarginSoftmax")
def _large_margin_softmax_grad(op, grad, _):
  '''The gradients for `LargeMarginSoftmax`.
  '''
  inputs_features = op.inputs[0]
  inputs_weights = op.inputs[1]
  inputs_labels = op.inputs[2]
  cur_lambda = op.outputs[1]
  #loss = op.outputs[0]
  margin_order = op.get_attr('margin_order')

  grads = op_module.large_margin_softmax_grad(inputs_features, inputs_weights, inputs_labels, grad, cur_lambda[0], margin_order)
  #print(grads)
  return [grads[0], grads[1], None, None]

class LargeMarginSoftmaxGradTest(tf.test.TestCase):
  def testLargeMarginSoftmaxGrad(self):
    with tf.device('/cpu:0'):
      large_margin_softmax = op_module.large_margin_softmax
      inputs_features = tf.constant(features, dtype=tf.float32)
      inputs_weights = tf.constant(weights, dtype=tf.float32)
      result = large_margin_softmax(inputs_features, inputs_weights, labels, 1, 4, 1000., 0.000025, 35., 0.)[0]
      with tf.Session() as sess:
        print('backprop large_margin_softmax in cpu:')
        print(tf.test.compute_gradient_error(inputs_features, [3, 4], result, [3, 5], delta=0.001, x_init_value=np.array(features)))
        print(tf.test.compute_gradient(inputs_features, [3, 4], result, [3, 5], delta=0.001, x_init_value=np.array(features)))

        print(tf.test.compute_gradient_error(inputs_weights, [5, 4], result, [3, 5], delta=0.001, x_init_value=np.array(weights)))
        print(tf.test.compute_gradient(inputs_weights, [5, 4], result, [3, 5], delta=0.001, x_init_value=np.array(weights)))
    with tf.device('/gpu:0'):
      large_margin_softmax = op_module.large_margin_softmax
      inputs_features = tf.constant(features, dtype=tf.float32)
      inputs_weights = tf.constant(weights, dtype=tf.float32)
      result = large_margin_softmax(inputs_features, inputs_weights, labels, 1, 4, 1000., 0.000025, 35., 0.)[0]

      with tf.Session(config = tf.ConfigProto(allow_soft_placement = True, log_device_placement = False)) as sess:
        print('backprop large_margin_softmax in gpu:')
        print(tf.test.compute_gradient_error(inputs_features, [3, 4], result, [3, 5], delta=0.001, x_init_value=np.array(features)))
        print(tf.test.compute_gradient(inputs_features, [3, 4], result, [3, 5], delta=0.001, x_init_value=np.array(features)))

        print(tf.test.compute_gradient_error(inputs_weights, [5, 4], result, [3, 5], delta=0.001, x_init_value=np.array(weights)))
        print(tf.test.compute_gradient(inputs_weights, [5, 4], result, [3, 5], delta=0.001, x_init_value=np.array(weights)))

if __name__ == "__main__":
  tf.test.main()
