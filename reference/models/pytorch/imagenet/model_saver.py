import torchvision.models as models
import torch

model = models.resnet50(pretrained=True)
model.eval()

batch = torch.randn((1, 3, 224, 224))
traced_model = torch.jit.trace(model, batch)
torch.jit.save(traced_model, 'resnet50.pt')
