var Jimp = require('jimp')
var Redis = require('ioredis')
var fs = require('fs')

const model_path = '../models/pytorch/imagenet/resnet50.pt'
const script_path = '../models/pytorch/imagenet/data_processing_script.txt'
const redis = new Redis({ parser: 'javascript' });
 
const json_labels = fs.readFileSync('../data/imagenet_classes.json')
const labels = JSON.parse(json_labels)

const image_width = 224;
const image_height = 224;

async function load_model() {
 

  const model = fs.readFileSync(model_path, {'flag': 'r'})
  const script = fs.readFileSync(script_path, {'flag': 'r'})
  
  redis.call('AI.MODELSET', 'imagenet_model', 'TORCH', 'CPU', "BLOB", model)
  redis.call('AI.SCRIPTSET', 'imagenet_script', 'CPU', "SOURCE", script)
}

async function run(filename) {
  let image = await Jimp.read(filename);
  let input_image = image.cover(image_width, image_height);
  let buffer = Buffer.from(input_image.bitmap.data);

  redis.call('AI.TENSORSET', 'image1',
             'UINT8', image_height, image_width, 4,
             'BLOB', buffer)
 
  redis.call('AI.SCRIPTRUN', 'imagenet_script', 'pre_process_4ch',
             'INPUTS', 'image1', 'OUTPUTS', 'temp1')
 
  redis.call('AI.MODELRUN', 'imagenet_model',
             'INPUTS', 'temp1', 'OUTPUTS', 'temp2')
 
  redis.call('AI.SCRIPTRUN', 'imagenet_script', 'post_process',
             'INPUTS', 'temp2', 'OUTPUTS', 'out')
 
  let out = await redis.call('AI.TENSORGET', 'out', 'VALUES')
  
  let idx = out[2][0]

  console.log(idx, labels[idx])
}

exports.load_model = load_model
exports.run = run


load_model()
const img_path = '../data/cat.jpg'
run(img_path)
