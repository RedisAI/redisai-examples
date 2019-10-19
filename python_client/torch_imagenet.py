import json
import time
import redisai as rai
import ml2rt
from skimage import io
from cli import arguments

if arguments.gpu:
    device = rai.Device.gpu
else:
    device = rai.Device.cpu

con = rai.Client(host=arguments.host, port=arguments.port)

pt_model_path = '../models/pytorch/imagenet/resnet50.pt'
script_path = '../models/pytorch/imagenet/data_processing_script.txt'
img_path = '../data/cat.jpg'

class_idx = json.load(open("../data/imagenet_classes.json"))

image = io.imread(img_path)

pt_model = ml2rt.load_model(pt_model_path)
script = ml2rt.load_script(script_path)

out1 = con.modelset('imagenet_model', rai.Backend.torch, device, pt_model)
out2 = con.scriptset('imagenet_script', device, script)
a = time.time()
tensor = rai.BlobTensor.from_numpy(image)
out3 = con.tensorset('image', tensor)
out4 = con.scriptrun('imagenet_script', 'pre_process_3ch', 'image', 'temp1')
out5 = con.modelrun('imagenet_model', 'temp1', 'temp2')
out6 = con.scriptrun('imagenet_script', 'post_process', 'temp2', 'out')
final = con.tensorget('out')
ind = final[0]
print(ind, class_idx[str(ind)])
print(time.time() - a)
