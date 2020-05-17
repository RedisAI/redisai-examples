var redis = require('redis');
var redisai = require('redisai-js');
var fs = require("fs");

//
// (async () => {
//     const nativeClient = redis.createClient();
//     const aiclient = new redisai.Client(nativeClient);
//     const tensorA = new redisai.Tensor(redisai.Dtype.float32, [1, 2], [3, 5]);
//     const result = await aiclient.tensorset("tensorA", tensorA);
//
//     // AI.TENSORSET result: OK
//     console.log(`AI.TENSORSET result: ${result}`)
//
//     const tensorGetReply = await aiclient.tensorget("tensorA");
//
//     // AI.TENSORGET reply: datatype FLOAT shape [1,2] , data [3,5]
//     console.log(`AI.TENSORGET reply: datatype ${tensorGetReply.dtype} shape [${tensorGetReply.shape}] , data [${tensorGetReply.data}]`);
//
//     await aiclient.end();
// })();
//
//
// (async () => {
//     const nativeClient = redis.createClient();
//     const aiclient = new redisai.Client(nativeClient);
//     const tensorA = new redisai.Tensor(redisai.Dtype.float32, [1, 2], [2, 3]);
//     const tensorB = new redisai.Tensor(redisai.Dtype.float32, [1, 2], [3, 5]);
//     const result_tA = await aiclient.tensorset("tA", tensorA);
//     const result_tB = await aiclient.tensorset("tB", tensorB);
//
//     const model_blob = fs.readFileSync("./../models/tensorflow/simple/graph.pb");
//     // AI.TENSORSET tA result: OK
//     console.log(`AI.TENSORSET tA result: ${result_tA}`)
//     // AI.TENSORSET tB result: OK
//     console.log(`AI.TENSORSET tB result: ${result_tB}`)
//
//     const mymodel = new redisai.Model(redisai.Backend.TF, "CPU", ["a", "b"], ["c"], model_blob);
//
//     const result_modelSet = await aiclient.modelset("mymodel", mymodel);
//
//     // AI.MODELSET result: OK
//     console.log(`AI.MODELSET result: ${result_modelSet}`)
//
//     const result_modelRun = await aiclient.modelrun("mymodel", ["tA", "tB"], ["tC"]);
//
//     console.log(`AI.MODELRUN result: ${result_modelRun}`)
//     const tensorC = await aiclient.tensorget("tC");
//
//     // AI.TENSORGET tC reply: datatype FLOAT shape [1,2] , data [6,15]
//     console.log(`AI.TENSORGET tC reply: datatype ${tensorC.dtype} shape [${tensorC.shape}] , data [${tensorC.data}]`);
//
//     await aiclient.end();
// })();


(async () => {
    const nativeClient = redis.createClient();
    const aiclient = new redisai.Client(nativeClient);


    const model_blob = fs.readFileSync("./../models/tensorflow/tinyyolo/tiny-yolo-voc.pb");

    const mobilenet = new redisai.Model(redisai.Backend.TF, "CPU", ["input"], ["output"], model_blob);

    const result_modelSet = await aiclient.modelset("yolo:model", mobilenet);

    // AI.MODELSET result: OK
    console.log(`AI.MODELSET yolo:model result: ${result_modelSet}`)

    await aiclient.end();
})();

//
// (async () => {
//     const nativeClient = redis.createClient();
//     const aiclient = new redisai.Client(nativeClient);
//     const tensorA = new redisai.Tensor(redisai.Dtype.float32, [1, 2], [2, 3]);
//     const tensorB = new redisai.Tensor(redisai.Dtype.float32, [1, 2], [3, 5]);
//     const script_str = 'def bar(a, b):\n    return a + b\n';
//
//     const result_tA = await aiclient.tensorset("tA", tensorA);
//     const result_tB = await aiclient.tensorset("tB", tensorB);
//     // AI.TENSORSET tA result: OK
//     console.log(`AI.TENSORSET tA result: ${result_tA}`)
//     // AI.TENSORSET tB result: OK
//     console.log(`AI.TENSORSET tB result: ${result_tB}`)
//
//     const myscript = new redisai.Script("CPU", script_str);
//
//     const result_scriptSet = await aiclient.scriptset("myscript", myscript);
//
//     // AI.SCRIPTSET result: OK
//     console.log(`AI.SCRIPTSET result: ${result_scriptSet}`)
//
//     const result_scriptRun = await aiclient.scriptrun("myscript", "bar",["tA", "tB"], ["tD"]);
//
//     console.log(`AI.SCRIPTRUN result: ${result_scriptRun}`)
//     const tensorD = await aiclient.tensorget("tD");
//
//     // AI.TENSORGET tD reply: datatype FLOAT shape [1,2] , data [5,8]
//     console.log(`AI.TENSORGET tD reply: datatype ${tensorD.dtype} shape [${tensorD.shape}] , data [${tensorD.data}]`);
//
//     await aiclient.end();
// })();
//
