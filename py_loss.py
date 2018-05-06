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
import numpy as np
import tensorflow as tf
import math

features = [[0.1, 0.2, -0.3, -0.4], [-1.1, -1.2, 1.3, 1.4], [2.1, 2.2, -2.3, -2.4]]
labels = [3, 2, 1]

def constant_xavier_initializer(shape, dtype=tf.float32, uniform=True):
    """Initializer function."""
    if not dtype.is_floating:
      raise TypeError('Cannot create initializer for non-floating point type.')
    # Estimating fan_in and fan_out is not possible to do perfectly, but we try.
    # This is the right thing for matrix multiply and convolutions.
    if shape:
      fan_in = float(shape[-2]) if len(shape) > 1 else float(shape[-1])
      fan_out = float(shape[-1])
    else:
      fan_in = 1.0
      fan_out = 1.0
    for dim in shape[:-2]:
      fan_in *= float(dim)
      fan_out *= float(dim)

    # Average number of inputs and output connections.
    n = (fan_in + fan_out) / 2.0
    if uniform:
      # To get stddev = math.sqrt(factor / n) need to adjust for uniform.
      limit = math.sqrt(3.0 * 1.0 / n)
      return tf.random_uniform(shape, -limit, limit, dtype, seed=None)
    else:
      # To get stddev = math.sqrt(factor / n) need to adjust for truncated.
      trunc_stddev = math.sqrt(1.3 * 1.0 / n)
      return tf.truncated_normal(shape, 0.0, trunc_stddev, dtype, seed=None)

def CosineFaceLoss(features, labels, embedding_dim, num_classes, scale=30., margin=0.35, scope=None):
    with tf.variable_scope(scope, "CosineFaceLoss", [features, labels]):
        var_weights = tf.Variable(constant_xavier_initializer([num_classes, embedding_dim]), name='weights')
        normed_weights = tf.nn.l2_normalize(var_weights, 1, 1e-10, name='weights_norm')
        normed_features = tf.nn.l2_normalize(features, 1, 1e-10, name='features_norm')

        cosine = tf.matmul(normed_features, normed_weights, transpose_a=False, transpose_b=True)

        cosine = tf.clip_by_value(cosine, -1, 1, name='cosine_clip') - margin * tf.one_hot(labels, num_classes, on_value=1., off_value=0., axis=-1, dtype=tf.float32)

        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                            logits=scale * cosine), name='cosine_loss')

def ArcFaceLoss(features, labels, embedding_dim, num_classes, scale=64., margin=0.5, easy_margin=True, scope=None):
    '''
    margin should in range [0, pi/2)
    '''
    with tf.variable_scope(scope, "ArcFaceLoss", [features, labels]):
        cos_m = math.cos(margin)
        sin_m = math.sin(margin)
        mm = math.sin(math.pi - margin) * margin
        threshold = math.cos(math.pi - margin)

        var_weights = tf.Variable(constant_xavier_initializer([num_classes, embedding_dim]), name='weights')
        normed_weights = tf.nn.l2_normalize(var_weights, 1, 1e-10, name='weights_norm')
        normed_features = tf.nn.l2_normalize(features, 1, 1e-10, name='features_norm')

        cosine = tf.matmul(normed_features, normed_weights, transpose_a=False, transpose_b=True)
        one_hot_mask = tf.one_hot(labels, num_classes, on_value=1., off_value=0., axis=-1, dtype=tf.float32)

        cosine_theta_2 = tf.pow(cosine, 2., name='cosine_theta_2')
        sine_theta = tf.pow(1. - cosine_theta_2, .5, name='sine_theta')

        cosine_theta_m = scale * (cos_m * cosine - sin_m * sine_theta) * one_hot_mask

        if easy_margin:
            clip_mask = tf.to_float(cosine >= 0.) * scale * cosine * one_hot_mask
        else:
            clip_mask = tf.to_float(cosine >= threshold) * scale * mm * one_hot_mask

        cosine = scale * cosine * (1. - one_hot_mask) + tf.where(clip_mask > 0., cosine_theta_m, clip_mask)

        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                            logits=cosine), name='arc_loss')

def FocalLoss(features, labels, num_classes, gamma=1.0, scope=None):
    with tf.variable_scope(scope, "FocalLoss", [features, labels]):
        one_hot = tf.one_hot(labels, num_classes, on_value=1., off_value=0., dtype=tf.float32)
        prob = tf.nn.softmax(features)
        return tf.reduce_mean(tf.reduce_sum(one_hot * (0. - tf.pow(1. - prob, gamma) * tf.nn.log_softmax(features)), axis=-1), name='focal_loss')

def test_cosine_loss():
    loss = CosineFaceLoss(tf.constant(features), tf.constant(labels), 4, 5)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print('cosine_loss:', sess.run(loss))

def test_arc_loss():
    loss = ArcFaceLoss(tf.constant(features), tf.constant(labels), 4, 5)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print('arc_loss:', sess.run(loss))

def test_focal_loss():
    loss = FocalLoss(tf.constant(features), tf.constant(labels), 4)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    print('focal_loss:', sess.run(loss))

if __name__ == "__main__":
    test_cosine_loss()
    test_arc_loss()
    test_focal_loss()
