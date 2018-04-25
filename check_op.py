#coding=utf-8
import numpy as np

import sys,os

caffe_root = '/media/rs/7A0EE8880EE83EAF/Detections/Caffe_Dev/'
sys.path.insert(0, caffe_root + 'python')
import caffe
os.chdir(caffe_root)

net_file='/media/rs/7A0EE8880EE83EAF/Detections/imgcls/codes/cpp/L_SoftMax/lsoftmax.prototxt'

caffe.set_mode_cpu()
net = caffe.Net('/media/rs/7A0EE8880EE83EAF/Detections/imgcls/codes/cpp/L_SoftMax/caffe/mnist/model/mnist_train_test.prototxt', caffe.TRAIN)
print(net.blobs.keys())


net.blobs['input_data'].data[...] = np.array([[0.1, 0.2, -0.3, -0.4], [-1.1, -1.2, 1.3, 1.4], [2.1, 2.2, -2.3, -2.4]])
net.blobs['label'].data[...] = np.array([3, 2, 1])
predict = net.forward()

print(predict)
#print(np.argmax(predict['prob'],axis=1))
