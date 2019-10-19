import json
import time
import redisai as rai
from ml2rt import load_model, load_script
from skimage import io
from cli import arguments

if arguments.gpu:
    device = rai.Device.gpu
else:
    device = rai.Device.cpu

con = rai.Client(host=arguments.host, port=arguments.port)

tf_model_path = '../models/tensorflow/imagenet/resnet50.pb'
script_path = '../models/tensorflow/imagenet/data_processing_script.txt'
img_path = '../data/cat.jpg'

class_idx = json.load(open("../data/imagenet_classes.json"))

image = io.imread(img_path)

tf_model = load_model(tf_model_path)
script = load_script(script_path)


out1 = con.modelset(
    'imagenet_model', rai.Backend.tf, device,
    inputs=['images'], outputs=['output'], data=tf_model)
out2 = con.scriptset('imagenet_script', device, script)
a = time.time()
tensor = rai.BlobTensor.from_numpy(image)
con.tensorset('image', tensor)
out4 = con.scriptrun('imagenet_script', 'pre_process_3ch', 'image', 'temp1')
out5 = con.modelrun('imagenet_model', 'temp1', 'temp2')
out6 = con.scriptrun('imagenet_script', 'post_process', 'temp2', 'out')
final = con.tensorget('out', as_type=rai.BlobTensor)
ind = final.to_numpy().item()
print(ind, class_idx[str(ind)])
print(time.time() - a)
