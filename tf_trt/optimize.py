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


import tensorflow as tf
import tensorflow_hub as hub

var_converter = tf.compat.v1.graph_util.convert_variables_to_constants
url = 'https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1'
images = tf.placeholder(tf.float32, shape=(1, 224, 224, 3), name='images')
module = hub.Module(url)
print(module.get_signature_names())
print(module.get_output_info_dict())
logits = module(images)
logits = tf.identity(logits, 'output')
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer()])
    graph_def = sess.graph_def

    # clearing device information
    for node in graph_def.node:
        node.device = ""
    frozen = tf.graph_util.convert_variables_to_constants(sess, graph_def, ['output'])
    trt_graph = trt.create_inference_graph(
    input_graph_def=frozen,
    outputs=['output']
    )

# DATA_DIR = '../models/tensorflow/imagenet/'
# MODEL = 'resnet50'
# TRT_SUFFIX = '_fp16_trt'

# output_names = ['output']

# print("------------- Load frozen graph from disk -------------")
# with tf.gfile.GFile(DATA_DIR + MODEL + '.pb', "rb") as f:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(f.read())

# print("------------- Optimize the model with TensorRT -------------")
#     # max_batch_size=1,
#     # max_workspace_size_bytes=1 << 26,
#     # precision_mode='FP16',
#     # minimum_segment_size=2
# trt_graph = trt.create_inference_graph(
#     input_graph_def=graph_def,
#     outputs=output_names

# )

# print("------------- Write optimized model to the file -------------")
# with open(DATA_DIR + MODEL + TRT_SUFFIX + '.pb', 'wb') as f:
#     f.write(trt_graph.SerializeToString())

# print("------------- DONE! -------------")