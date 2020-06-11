import tensorflow as tf
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt

tf_model_path = '../models/tensorflow/imagenet/resnet50.pb'
tf_trt_model_path = '../models/tensorflow-trt/imagenet/tf-trt-resnet50.pb'

def load_model(path):
    with tf.gfile.GFile(path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

graph_def = load_model(tf_model_path)

converter = trt.TrtGraphConverter(input_graph_def=graph_def)
graph_def = converter.convert()
converter.save('../models/tensorflow-trt/imagenet/')
