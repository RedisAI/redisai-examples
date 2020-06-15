# on aws dl ami source activate tensorflow_p36

import tensorflow as tf
import tensorflow_hub as hub
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.client import device_lib

tf_trt_model_path = '../../models/tensorflow/resnet50/resnet50_fp16_trt.pb'

var_converter = tf.compat.v1.graph_util.convert_variables_to_constants
url = 'https://tfhub.dev/google/imagenet/resnet_v2_50/classification/1'

gpu_available = tf.test.is_gpu_available(
    cuda_only=True, min_cuda_compute_capability=None
)

if gpu_available is False:
    print("No CUDA GPUs found. Exiting...")
    sys.exit(1)

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

    frozen_optimized = converter.convert()
    directory = os.path.dirname(tf_trt_model_path)
    file = os.path.basename(tf_trt_model_path)
    tf.io.write_graph(frozen_optimized, directory, file, as_text=False)