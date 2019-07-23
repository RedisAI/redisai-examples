var Redis = require('ioredis');
var fs = require('fs')

const all_chars = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ \t\n\r\x0b\x0c'

const model_path = "../models/pytorch/charrnn/charrnn_pipeline.pt"
const redis = new Redis({ parser: 'javascript' });

function int2str(int_data) {
  let out_str = '';
  for (let i=0; i < int_data.length; i++) {
    out_str += all_chars.charAt(int_data[i])
  }
  return out_str
}

const hidden_size = 300
const n_layers = 2
const batch_size = 1

async function load_model() {
  const model = fs.readFileSync(model_path, {'flag': 'r'})
  redis.call('AI.MODELSET', 'char_rnn', 'TORCH', 'CPU', model)
}

async function run(prime) {
  let hidden = new Float32Array(n_layers * batch_size * hidden_size)
  hidden.fill(0.0)

  redis.call('AI.TENSORSET', 'hidden', 'FLOAT',
             n_layers, batch_size, hidden_size, 'BLOB', Buffer.from(hidden.buffer))

  redis.call('AI.TENSORSET', 'prime', 'INT64', 1, 'VALUES', prime)

  redis.call('AI.MODELRUN', 'char_rnn', 'INPUTS', 'prime', 'hidden', 'OUTPUTS', 'out')

  let out = await redis.call('AI.TENSORGET', 'out', 'VALUES')

  let text = int2str(out[2]);

  console.log(text);
}

exports.load_model = load_model
exports.run = run


load_model()
const prime = 1
run(prime)
