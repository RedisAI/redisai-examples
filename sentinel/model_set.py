import redisai as rai

con = rai.Client(host='159.65.150.75', port=6379, db=0)

pt_model_path = '../models/imagenet/pytorch/resnet50.pt'
script_path = '../models/imagenet/pytorch/data_processing_script.txt'

pt_model = rai.load_model(pt_model_path)
script = rai.load_script(script_path)

out1 = con.modelset('imagenet_model', rai.Backend.torch, rai.Device.cpu, pt_model)
out2 = con.scriptset('imagenet_script', rai.Device.cpu, script)
