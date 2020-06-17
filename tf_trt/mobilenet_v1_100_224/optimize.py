from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.python.compiler.tensorrt import trt_convert as trt
import logging
import sys
import tensorflow_hub as hub
import tensorflow as tf
import os

#  restricting execution to a specific device or set of devices for debugging and testing
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# on aws dl ami source activate tensorflow_p36


compiled_version = trt.wrap_py_utils.get_linked_tensorrt_version()
loaded_version = trt.wrap_py_utils.get_loaded_tensorrt_version()
print("$$$$$$$$$$$$$$$$$$$$$$$$ Linked TensorRT version: %s" %
      str(compiled_version))
print("$$$$$$$$$$$$$$$$$$$$$$$$ Loaded TensorRT version: %s" %
      str(loaded_version))

trt._check_trt_version_compatibility()

logging.getLogger("tensorflow").setLevel(logging.ERROR)

tf_trt_model_path = '../../models/tensorflow/mobilenet/mobilenet_v1_100_224_gpu_NxHxWxC_fp16_trt.pb'

gpu_available = tf.test.is_gpu_available(
    cuda_only=True, min_cuda_compute_capability=None
)

print("TensorFlow version: ", tf.__version__)
if gpu_available is False:
    print("No CUDA GPUs found. Exiting...")
    sys.exit(1)

model = tf.keras.Sequential([
    hub.KerasLayer(
        "https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/4")
])
model.build([None, 224, 224, 3])  # Batch input shape.

inputs = tf.keras.Input(batch_input_shape=(1,224,224,3))
config = model.get_config()

newOutputs = model(inputs)
newModel = tf.keras.Model(inputs,newOutputs)

tf.saved_model.save(newModel, 'mobilenet_v1_100_224_saved_model')

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(
    precision_mode=trt.TrtPrecisionMode.FP16)


print("Optimizing the model with TensorRT")

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir='mobilenet_v1_100_224_saved_model',
    conversion_params=conversion_params)

frozen_optimized = converter.convert()

converter.save(output_saved_model_dir='mobilenet_v1_100_224_gpu_NxHxWxC_fp16_trt')
print('Done Converting to TF-TRT FP16')


# Load model and get the concrete function.
model = tf.saved_model.load('mobilenet_v1_100_224_gpu_NxHxWxC_fp16_trt')
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

constantGraph = convert_variables_to_constants_v2(concrete_func)

directory = os.path.dirname(tf_trt_model_path)
file = os.path.basename(tf_trt_model_path)
tf.io.write_graph(constantGraph.graph.as_graph_def(), directory, file, as_text=False)
