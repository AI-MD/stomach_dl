# 필요한 import문

import io
import numpy as np
from torch import nn

import torch.utils.model_zoo as model_zoo
import torch.onnx
import torchsummary

# 미리 학습된 가중치를 읽어옵니다
#model_url = './result_1104_softentropy.pth'
model_url = './result_1018_bce.pth'
batch_size = 1


dev="cuda"

model = torch.load(model_url)
model.eval()

model.to(dev)

#model.set_swish(memory_efficient=False)

x= torch.randn(batch_size, 3, 224, 224, requires_grad=True)
x = x.to(dev)

torch_out = model(x)
torchsummary.summary(model, (3, 224, 224))
model.set_swish(memory_efficient=False)
torch.onnx.export(model, x,"./efficientnet_1018.onnx")

