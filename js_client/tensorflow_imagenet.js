var Jimp = require('jimp')
var fs = require('fs')
var redis = require('redis');
var redisai = require('redisai-js');

const model_path = '../models/tensorflow/imagenet/resnet50.pb'
const script_path = '../models/tensorflow/imagenet/data_processing_script.txt'
const json_labels = fs.readFileSync('../data/imagenet_classes.json')
const labels = JSON.parse(json_labels)

const image_width = 224;
const image_height = 224;

async function load_model() {
    const nativeClient = redis.createClient();
    const aiclient = new redisai.Client(nativeClient);

    const modelBlob = fs.readFileSync(model_path);
    const scriptString = fs.readFileSync(script_path, {'flag': 'r'}).toString();

    const imagenetModel = new redisai.Model(redisai.Backend.TF, "CPU", ["images"], ["output"], modelBlob);

    const result_modelSet = await aiclient.modelset("imagenet_model", imagenetModel);

    // AI.MODELSET result: OK
    console.log(`AI.MODELSET imagenet_model result: ${result_modelSet}`);


    const myscript = new redisai.Script("CPU", scriptString);

    const result_scriptSet = await aiclient.scriptset("imagenet_script", myscript);

    // AI.SCRIPTSET result: OK
    console.log(`AI.SCRIPTSET imagenet_script result: ${result_scriptSet}`);

    await aiclient.end(true);
}

async function run(filename) {

    const nativeClient = redis.createClient();
    const aiclient = new redisai.Client(nativeClient);

    let image = await Jimp.read(filename);
    let input_image = image.cover(image_width, image_height);
    let buffer = Buffer.from(input_image.bitmap.data);

    const tensor = new redisai.Tensor(redisai.Dtype.uint8, [image_height, image_width, 4], buffer);
    const result = await aiclient.tensorset("image1", tensor);

    // AI.TENSORSET image1 result: OK
    console.log(`AI.TENSORSET image1 result: ${result}`);

    const resultScriptRun = await aiclient.scriptrun("imagenet_script", "pre_process_4ch", ["image1"], ["temp1"]);

    // AI.SCRIPTRUN imagenet_script pre_process_4ch result: OK
    console.log(`AI.SCRIPTRUN imagenet_script pre_process_4ch result: ${resultScriptRun}`);

    const resultModelRun = await aiclient.modelrun("imagenet_model", ["temp1"], ["temp2"]);

    // AI.MODELRUN imagenet_model result: OK
    console.log(`AI.MODELRUN imagenet_model result: ${resultModelRun}`);

    const resultScriptRun2 = await aiclient.scriptrun("imagenet_script", "post_process", ["temp2"], ["out"]);

    // AI.SCRIPTRUN imagenet_script post_process result: OK
    console.log(`AI.SCRIPTRUN imagenet_script post_process result: ${resultScriptRun2}`);

    const tensorGetReply = await aiclient.tensorget("out");
    let idx = tensorGetReply.data
    console.log(idx, labels[idx.toString()])

    await aiclient.end();
}

load_model().then(
    () => {
        const img_path = '../data/cat.jpg'
        run(img_path)
    }
)



