import redis
from skimage import io
import json

r = redis.Redis(host='159.65.150.75', port=6379, db=0)


img_path = '../models/imagenet/data/cat.jpg'

class_idx = json.load(open("../models/imagenet/data/imagenet_classes.json"))

image = io.imread(img_path)


out3 = r.execute_command(
    'AI.TENSORSET', 'image', 'UINT8', *image.shape, 'BLOB', image.tobytes())
out4 = r.execute_command(
    'AI.SCRIPTRUN', 'imagenet_script', 'pre_process', 'INPUTS', 'image', 'OUTPUTS', 'temp1')
out5 = r.execute_command('AI.MODELRUN', 'imagenet_model', 'INPUTS', 'temp1', 'OUTPUTS', 'temp2')
out6 = r.execute_command(
    'AI.SCRIPTRUN', 'imagenet_script', 'post_process', 'INPUTS', 'temp2', 'OUTPUTS', 'out')
final = r.execute_command('AI.TENSORGET', 'out', 'VALUES')
ind = final[2][0]
print(ind, class_idx[str(ind)])
