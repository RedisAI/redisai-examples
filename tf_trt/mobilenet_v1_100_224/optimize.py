import tensorflow as tf
import tensorflow_hub as hub
import ml2rt
import sys
import logging
import os
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.client import device_lib

os.environ['CUDA_VISIBLE_DEVICES']='0'
# on aws dl ami source activate tensorflow_p36

print("TensorFlow version: ", tf.__version__)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
var_converter = tf.compat.v1.graph_util.convert_variables_to_constants

tf_trt_model_path = '../../models/tensorflow/mobilenet/mobilenet_v1_100_224_gpu_NxHxWxC_fp16_trt.pb'

url = 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/classification/3'

gpu_available = tf.test.is_gpu_available(
    cuda_only=True, min_cuda_compute_capability=None
)

if gpu_available is False:
    print("No CUDA GPUs found. Exiting...")
    sys.exit(1)

module = hub.Module(url)
batch_size = 1
height, width = hub.get_expected_image_size(module)
input_var = 'input'
output_var = 'MobilenetV1/Predictions/Reshape_1'
images = tf.placeholder(tf.float32, shape=(batch_size, height, width, 3), name=input_var)
print(module.get_signature_names())
print(module.get_output_info_dict())
logits = module(images)
logits = tf.identity(logits, output_var)



with tf.Session() as sess:
    sess.run([tf.global_variables_initializer()])
    graph_def = sess.graph_def

    # clearing device information
    for node in graph_def.node:
        node.device = ""
    frozen = tf.graph_util.convert_variables_to_constants(
        sess, graph_def, [output_var])

    print("Optimizing the model with TensorRT")
    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16,  input_graph_def=frozen )

    converter = trt.TrtGraphConverter(
        *conversion_params
    )

    frozen_optimized = converter.convert()
    directory = os.path.dirname(tf_trt_model_path)
    file = os.path.basename(tf_trt_model_path)
    tf.io.write_graph(frozen_optimized, directory, file, as_text=False)

