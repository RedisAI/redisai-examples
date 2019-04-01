import redis
from skimage import io
import json
import time

r = redis.Redis(host='localhost', port=6379, db=0)

pt_model_path = '../models/imagenet/pytorch/resnet50.pt'
script_path = '../models/imagenet/pytorch/data_processing_script.txt'
img_path = '../models/imagenet/data/cat.jpg'

class_idx = json.load(open("../models/imagenet/data/imagenet_classes.json"))

image = io.imread(img_path)

with open(pt_model_path, 'rb') as f:
    pt_model = f.read()

with open(script_path, 'rb') as f:
    script = f.read()

out1 = r.execute_command('AI.MODELSET', 'imagenet_model', 'TORCH', 'CPU', pt_model)
out2 = r.execute_command('AI.SCRIPTSET', 'imagenet_script', 'CPU', script)
a = time.time()
out3 = r.execute_command(
    'AI.TENSORSET', 'image', 'UINT8', *image.shape, 'BLOB', image.tobytes())
out4 = r.execute_command(
    'AI.SCRIPTRUN', 'imagenet_script', 'pre_process_3ch', 'INPUTS', 'image', 'OUTPUTS', 'temp1')
out5 = r.execute_command('AI.MODELRUN', 'imagenet_model', 'INPUTS', 'temp1', 'OUTPUTS', 'temp2')
out6 = r.execute_command(
    'AI.SCRIPTRUN', 'imagenet_script', 'post_process', 'INPUTS', 'temp2', 'OUTPUTS', 'out')
final = r.execute_command('AI.TENSORGET', 'out', 'VALUES')
ind = final[2][0]
print(ind, class_idx[str(ind)])
print(time.time() - a)
