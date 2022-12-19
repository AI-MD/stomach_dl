import torch
import onnx
onnx_model = onnx.load("stomach_model.onnx")
onnx.checker.check_model(onnx_model)
import onnxruntime
from pathlib import Path
ort_session = onnxruntime.InferenceSession("stomach_model.onnx")
import os
from util.utils import preprocessing
from torchvision import models, transforms
import numpy as np
input_size = (224, 224)

# Preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor()
])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()



import cv2 as cv



files=Path("./images_1128/S5/").resolve().glob('*.*')

images=list(files)
for num,img in enumerate(images):
    print(os.path.abspath(img))
    image = cv.imread(os.path.abspath(img))

    img = preprocessing(image)
    inputs = preprocess(img)
    inputs = inputs.unsqueeze(0).to(device)


    # for i in range(30):
    #     print(to_numpy(inputs)[0][0][0][i])

    #print(np.ravel(to_numpy(inputs),order='C')[0:30])


    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(inputs)}
    ort_outs = ort_session.run(None, ort_inputs)

    #print(ort_inputs.get('input.1').shape)


    ort_outs[0]= 1/(1 + np.exp(-ort_outs[0])) * 100


    print(ort_outs[0])
    print(np.max(ort_outs[0]),np.argmax(ort_outs[0]))


# ONNX 타임에서 계산된 결과값


#for i in range(224):
#    for j in range(224):
#        print(to_numpy(inputs)[0][0][j][i])



