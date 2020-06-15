import json
import time
import redisai as rai
from ml2rt import load_model, load_script
from skimage import io
from cli import arguments

if arguments.gpu:
    device = 'gpu'
else:
    device = 'cpu'

con = rai.Client(host=arguments.host, port=arguments.port)

tf_model_path = '../models/tensorflow/mobilenet/mobilenet_v1_100_224.pb'
script_path = '../models/tensorflow/resnet50/data_processing_script.txt'
img_path = '../data/cat.jpg'
input_var = 'input'
output_var = 'MobilenetV1/Predictions/Reshape_1'
class_idx = json.load(open("../data/imagenet_classes.json"))

image = io.imread(img_path)

tf_model = load_model(tf_model_path)
script = load_script(script_path)


out1 = con.modelset(
    'mobilenet_224', 'tf', device,
    inputs=[input_var], outputs=[output_var], data=tf_model)
out2 = con.scriptset('imagenet_script', device, script)
a = time.time()
con.tensorset('image', image)
out4 = con.scriptrun('imagenet_script', 'pre_process_3ch', 'image', 'temp1')
out5 = con.modelrun('mobilenet_224', 'temp1', 'temp2')
out6 = con.scriptrun('imagenet_script', 'post_process', 'temp2', 'out')
final = con.tensorget('out')
ind = final.item()
print(ind, class_idx[str(ind)])
print(time.time() - a)
