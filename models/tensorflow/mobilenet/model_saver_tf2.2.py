import argparse
import sys

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

model_name = 'tf2_mobilenet_v1_100_224'
batch_size = 1
number_channels = 3
width = 224
height = 224
input_var = 'input'
output_var = 'MobilenetV1/Predictions/Reshape_1'

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', action="store_true", default=False)
parser.add_argument('--input-shape', default="NxHxWxC", type=str)
args = parser.parse_args()
device = 'gpu' if args.gpu else 'cpu'

filename = '{model_name}_{device}_{input_shape}.pb'.format(
    model_name=model_name, device=device, input_shape=args.input_shape)

gpu_available = tf.test.is_gpu_available(
    cuda_only=True, min_cuda_compute_capability=None
)

if gpu_available is False and args.gpu:
    print("No CUDA GPUs found. Exiting...")
    sys.exit(1)

model = tf.keras.Sequential([
    hub.KerasLayer(
        "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/4")
])
model.build([batch_size, height, width, number_channels])  # Batch input shape.

inputs = tf.keras.Input(batch_input_shape=(batch_size, height, width, number_channels), name=input_var)

newOutputs = model(inputs)
model = tf.keras.Model(inputs, newOutputs)

full_model = tf.function(lambda x: model(x))

concrete_func = full_model.get_concrete_function(
    (tf.TensorSpec(inputs.shape, tf.float32, name=inputs.name)))

constantGraph = convert_variables_to_constants_v2(concrete_func)

tf.io.write_graph(constantGraph.graph.as_graph_def(), ".", filename, as_text=False)
