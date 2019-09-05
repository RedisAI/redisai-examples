import redisai as rai
import ml2rt
import numpy as np
from PIL import Image
from PIL import ImageDraw

from cli import arguments

IMG_SIZE = 416
labels20 = {
    1: "aeroplane",
    2: "bicycle",
    3: "bird",
    4: "boat",
    5: "bottle",
    6: "bus",
    7: "car",
    8: "cat",
    9: "chair",
    10: "cow",
    11: "diningtable",
    12: "dog",
    13: "horse",
    14: "motorbike",
    15: "person",
    16: "pottedplant",
    17: "sheep",
    18: "sofa",
    19: "train",
    20: "tvmonitor"}


if arguments.gpu:
    device = rai.Device.gpu
else:
    device = rai.Device.cpu

con = rai.Client(host=arguments.host, port=arguments.port)
model = ml2rt.load_model('../models/tensorflow/tinyyolo/tinyyolo.pb')
script = ml2rt.load_script('../models/tensorflow/tinyyolo/yolo_boxes_script.py')

con.modelset('yolo', rai.Backend.tf, device, model, ['input'], ['output'])
con.scriptset('yolo-post', device, script)

img_jpg = Image.open('../data/sample_dog_416.jpg')
# normalize
img = np.array(img_jpg).astype(np.float32)
img = np.expand_dims(img, axis=0)
img /= 256.0

tensor = rai.BlobTensor.from_numpy(img)
con.tensorset('in', tensor)
con.modelrun('yolo', 'in', 'out')
con.scriptrun('yolo-post', 'boxes_from_tf', input='out', output='boxes')
boxes = con.tensorget('boxes', as_type=rai.BlobTensor).to_numpy()

n_boxes = 0
for box in boxes[0]:
    if box[4] == 0.0:
        continue

    n_boxes += 1
    x1 = img.shape[0] * (box[0] - 0.5 * box[2])
    x2 = img.shape[0] * (box[0] + 0.5 * box[2])
    y1 = img.shape[1] * (box[1] - 0.5 * box[3])
    y2 = img.shape[1] * (box[1] + 0.5 * box[3])

    label = labels20[int(box[-1])]

    draw = ImageDraw.Draw(img_jpg)
    draw.rectangle(((x1, y1), (x2, y2)), outline='green')
    draw.text((x1, y1), label)
    img_jpg.save('out.jpg', "JPEG")
print('Number of valid boxes: ', n_boxes)
