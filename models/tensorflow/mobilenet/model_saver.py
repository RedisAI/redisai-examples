import tensorflow as tf
import tensorflow_hub as hub
import ml2rt

var_converter = tf.compat.v1.graph_util.convert_variables_to_constants

url = 'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/quantops/classification/3'
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
    ml2rt.save_tensorflow(sess, 'mobilenet_v1_100_224.pb', output=[output_var])
