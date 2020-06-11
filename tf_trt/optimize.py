## on aws dl ami source activate tensorflow_p36

import tensorflow as tf
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.client import device_lib 

tf_model_path = '../models/tensorflow/imagenet/resnet50.pb'
tf_trt_model_path = '../models/tensorflow-trt/imagenet/tf-trt-resnet50.pb'

def check_tensor_core_gpu_present(): 
    result = False
    local_device_protos = device_lib.list_local_devices() 
    for line in local_device_protos: 
        if "compute capability" in str(line): 
            compute_capability = float(line.physical_device_desc.split("compute capability: ")[-1]) 
            if compute_capability>=7.0: 
                result = True 
    
    return result 
            
print("Tensor Core GPU Present:", check_tensor_core_gpu_present()) 

# config = tf.ConfigProto() 
# config.gpu_options.allow_growth = True 
# config.allow_soft_placement = True 
# # graph = tf.Graph().ParseFromString

def load_model(path):
    with tf.gfile.GFile(path, "rb") as f:
        print(f)
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")
    return graph

graph_def = load_model(tf_model_path)

converter = trt.TrtGraphConverter(input_graph_def=graph_def)
graph_def = converter.convert()
converter.save('../models/tensorflow-trt/imagenet/')
