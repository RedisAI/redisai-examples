# on aws dl ami source activate tensorflow_p36

import tensorflow as tf
import tensorflow_hub as hub
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.client import device_lib

tf_trt_model_path = '../models/tensorflow/imagenet/resnet50_fp16_trt.pb'

var_converter = tf.compat.v1.graph_util.convert_variables_to_constants
url = 'https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1'

images = tf.placeholder(tf.float32, shape=(1, 224, 224, 3), name='images')
module = hub.Module(url)
logits = module(images)
logits = tf.identity(logits, 'output')

with tf.Session() as sess:
    sess.run([tf.global_variables_initializer()])
    graph_def = sess.graph_def

    # clearing device information
    for node in graph_def.node:
        node.device = ""
    frozen = tf.graph_util.convert_variables_to_constants(
        sess, graph_def, ['output'])

    print("Optimizing the model with TensorRT")

    converter = trt.TrtGraphConverter(
        input_graph_def=frozen,
        nodes_blacklist=['output'],
        precision_mode='FP16',
    )

    frozen_graph = converter.convert()
    directory = os.path.dirname(tf_trt_model_path)
    file = os.path.basename(tf_trt_model_path)
    tf.io.write_graph(frozen_graph, directory, file, as_text=False)

    # print("------------- Write optimized model to the file -------------")
    # with open(tf_trt_model_path, 'wb') as f:
    #     f.write(trt_graph.SerializeToString())

    # trt_graph.

    # print("------------- DONE! -------------")
