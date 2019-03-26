import redis

r = redis.Redis(host='159.65.150.75', port=6379, db=0)

pt_model_path = '../models/imagenet/pytorch/resnet50.pt'
script_path = '../models/imagenet/pytorch/data_processing_script.txt'


with open(pt_model_path, 'rb') as f:
    pt_model = f.read()

with open(script_path, 'rb') as f:
    script = f.read()

out1 = r.execute_command('AI.MODELSET', 'imagenet_model', 'TORCH', 'CPU', pt_model)
out2 = r.execute_command('AI.SCRIPTSET', 'imagenet_script', 'CPU', script)
