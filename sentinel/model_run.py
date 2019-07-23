import redisai as rai
from skimage import io
import json

con = rai.Client(host='159.65.150.75', port=6379, db=0)


img_path = '../models/imagenet/data/cat.jpg'

class_idx = json.load(open("../models/imagenet/data/imagenet_classes.json"))

image = io.imread(img_path)

tensor = rai.BlobTensor.from_numpy(image)
out3 = con.tensorset('image', tensor)
out4 = con.scriptrun('imagenet_script', 'pre_process', 'image', 'temp1')
out5 = con.modelrun('imagenet_model', 'temp1', 'temp2')
out6 = con.scriptrun('imagenet_script', 'post_process', 'temp2', 'out')
final = con.tensorget('out')
ind = final.value[0]
print(ind, class_idx[str(ind)])
