import tensorflow as tf
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt

tf_model_path = '../models/tensorflow/imagenet/resnet50.pb'
tf_trt_model_path = '../models/tensorflow-trt/imagenet/tf-trt-resnet50.pb'

graph_def = tf.GraphDef()
with tf.gfile.GFile(tf_model_path, 'rb') as f:
    graph_def.ParseFromString(f.read())

converter = trt.TrtGraphConverter(input_graph_def=graph_def)
graph_def = converter.convert()
converter.save('../models/tensorflow-trt/imagenet/')
