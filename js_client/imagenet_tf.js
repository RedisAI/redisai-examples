var Jimp = require('jimp')
var Redis = require('ioredis')
var fs = require('fs')

const model_path = '../models/imagenet/tensorflow/resnet50.pb'
const script_path = '../models/imagenet/tensorflow/data_processing_script.txt'
 
const json_labels = fs.readFileSync('../models/imagenet/data/imagenet_classes.json')
const labels = JSON.parse(json_labels)

const image_width = 224;
const image_height = 224;

async function load_model() {
 
  let redis = new Redis({ parser: 'javascript' });

  const model = fs.readFileSync(model_path, {'flag': 'r'})
  const script = fs.readFileSync(script_path, {'flag': 'r'})
  
  redis.call('AI.MODELSET', 'imagenet_model', 'TF', 'CPU', 
             'INPUTS', 'images', 'OUTPUTS', 'output', model)
  redis.call('AI.SCRIPTSET', 'imagenet_script', 'CPU', script)
}

async function run(filename) {
 
  let redis = new Redis({ parser: 'javascript' });

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

  console.log(idx, labels[idx.toString()])
}

exports.load_model = load_model
exports.run = run

/*
load_model()

const img_path = 'imagenet/data/cat.jpg'
run(img_path)
*/
