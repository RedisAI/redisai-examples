import torch
from ml2rt import save_torch

debug = True
device = 'cpu' if not torch.cuda.is_available() else 'cuda'
traced_generator = torch.jit.load('charrnn_model.pt', device)


class ScriptModuleWrapper(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.traced_generator = traced_generator

    @torch.jit.script_method
    def forward(self, inputs, hidden):
        i = 0
        prediction_seq_length = 200
        out = torch.zeros(prediction_seq_length, dtype=torch.long)
        while i < prediction_seq_length:
            output, hidden = self.traced_generator(inputs, hidden)
            inputs = self.post_processing(output)
            out[i] = inputs
            i += 1
        return out

    @torch.jit.script_method
    def post_processing(self, output):
        output_dist = output.squeeze().div(0.8).exp()
        prob = torch.multinomial(output_dist, 1)[0]
        return prob


filename = 'charrnn_pipeline.pt'
print('Saving pipeline to {}'.format(filename))
net = ScriptModuleWrapper()
save_torch(net, filename)  # equivalent to net.save(filename)
