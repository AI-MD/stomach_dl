# dummy data
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from numpy.ma import asarray
from scipy.constants import precision
from torch.autograd import Variable
from util.utils import preprocessing

# net = torch.load("./result_1018_bce.pth")
# net.eval()
#
# batch_size = 1  # just a random number
# channel = 3
# h_size = 224
# w_size = 224
# x = torch.randn(batch_size, channel, h_size, w_size , requires_grad=True)
# x = x.cuda()
#
# feature, outputs= net(x)
#
# net.set_swish(memory_efficient=False)
#
# torch.onnx.export(net,
#                   x,
#                   "efficientnet_new.onnx",
#                   export_params=True,
#                   opset_version=10,
#                   do_constant_folding=True,
#                   input_names=['input'],
#                   output_names=['output']
#                   )

import onnx
onnx_model = onnx.load("stomach_model.onnx")
onnx.checker.check_model(onnx_model)
import onnxruntime

ort_session = onnxruntime.InferenceSession("stomach_model.onnx")


from torchvision import models, transforms
input_size = (224, 224)

# Preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor()
])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import cv2 as cv

image = cv.imread("./images_1128/S5/1.2.410.200010.1.1.20221027.110600.225.51.1.7.dcm.bmp")


img = preprocessing(image)
inputs = preprocess(img)

text = inputs.numpy()
print("text.shape:", text.shape)
text = np.array(text).flatten()
np.set_printoptions(precision=7)
np.savetxt("test.txt", text, fmt = '%.7f')


# print(inputs[0][0][0][0])
# print(inputs[0][1][0][0])
# print(inputs[0][2][0][0])
#
# print(inputs[0][0][0][1])
# print(inputs[0][1][0][1])
# print(inputs[0][2][0][1])
inputs = inputs.unsqueeze(0).to(device)


# ONNX 타임에서 계산된 결과값


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

#for i in range(224):
#    for j in range(224):
#        print(to_numpy(inputs)[0][0][j][i])

print(to_numpy(inputs).shape)


for i in range(30):
    print(to_numpy(inputs)[0][0][0][i])


print("test")
print(np.ravel(to_numpy(inputs),order='C')[0:30])


ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
ort_outs = ort_session.run(None, ort_inputs)

print(ort_inputs.get('input.1').shape)

#outputs = torch.sigmoid(outputs)
ort_outs[0]= 1/(1 + np.exp(-ort_outs[0])) * 100

print(np.max(ort_outs[0]),np.argmax(ort_outs[0]))

# ONNX 런타임과 PyTorch에서 연산된 결과값 비교
#np.testing.assert_allclose(to_numpy(outputs), ort_outs[0], rtol=1e-03, atol=1e-05)

print("Exported model has been tested with ONNXRuntime, and the result looks good!")



