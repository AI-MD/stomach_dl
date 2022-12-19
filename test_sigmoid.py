import numpy as np
import sys, random
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from util.utils import preprocessing,softmax
import cv2 as cv
import os
import time
from util.gradcam import GradCAM, GradCAMpp
from torchvision.utils import make_grid, save_image
from util.utils import visualize_cam,Normalize
import torch.nn.functional as F
from torchsummary import summary

# Paths for image directory and model
IMDIR=sys.argv[1]
model_name=sys.argv[2]
MODEL=sys.argv[3]

input_size=(int(sys.argv[4]), int(sys.argv[4]))
# MODEL= "models/se_resnet18_center_softmax_64_30_adam_cosine_re_5.pth"
# Load the model for testing
model = torch.load(MODEL)
model.eval()

# Class labels for prediction
# class_namdes=['S1', 'S2', 'S3', 'S4', 'S5']

check_class  = { 'S1': False,
                 'S2': False,
                 'S3': False,
                 'S4': False,
                 'S5': False}

class_names=['C','D1','D2','E','S1', 'S2', 'S3', 'S4', 'S5', 'S6']


# Retreive 9 random images from directory
files=Path(IMDIR).resolve().glob('*.*')

images=list(files)

# Configure plots
fig = plt.figure(figsize=(12,12))
rows,cols = 9,8

# Preprocessing transformations
preprocess=transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor()
            
           ])

# Enable gpu mode, if cuda available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#grad_cam을 위한 작업

cam_dict = dict()


if model_name=='resnet':

    resnet_model_dict = dict(type='resnet', arch=model, layer_name='layer4', input_size=input_size)
    resnet_gradcam = GradCAM(resnet_model_dict, True)
    resnet_gradcampp = GradCAMpp(resnet_model_dict, True)

    cam_dict['resnet'] = [resnet_gradcam, resnet_gradcampp]

    gradcam, gradcam_pp = cam_dict['resnet']
elif model_name=='densenet':
    densenet_model_dict = dict(type=model_name, arch=model, layer_name='features_norm5', input_size=input_size)
    densenet_gradcam = GradCAM(densenet_model_dict, True)
    densenet_gradcampp = GradCAMpp(densenet_model_dict, True)

    cam_dict['densenet'] = [densenet_gradcam, densenet_gradcampp]

    gradcam, gradcam_pp = cam_dict['densenet']
else:
    efficientnet_model_dict = dict(type=model_name, arch=model, layer_name='_blocks.15', input_size=input_size)
    efficientnet_gradcam = GradCAM(efficientnet_model_dict, True)
    efficientnet_gradcampp = GradCAMpp(efficientnet_model_dict, True)

    cam_dict['efficientnet'] = [efficientnet_gradcam, efficientnet_gradcampp]

    gradcam, gradcam_pp = cam_dict['efficientnet']

grad_images = []



for num,img in enumerate(images):

    image = cv.imread(os.path.abspath(img))
    base = os.path.basename(os.path.abspath(img))
    img_proc = preprocessing(image)
    inputs = preprocess(img_proc).unsqueeze(0).to(device)

    torch_img = torch.from_numpy(np.asarray(img_proc)).permute(2, 0, 1).unsqueeze(0).float().div(255).cuda()
    torch_img = F.upsample(torch_img, size=input_size, mode='bilinear', align_corners=False)

    mask, logits = gradcam(inputs)
    mask = mask.cpu().detach().numpy()
    heatmap, result = visualize_cam(mask, torch_img)

    print(logits)

    logits = torch.sigmoid(logits)
    _, predicted = torch.max(logits.data, 1)
    print(class_names[predicted])

    logits_numpy = logits.cpu().detach().numpy().squeeze()

    print("------------")
    for val in logits_numpy:
        print("{:.5f}".format(val))
    print("------------")

    dir = os.path.dirname(os.path.abspath(img))
    base_name= os.path.basename(dir)
    os.makedirs(base_name, exist_ok=True)
    print(base_name)
    cv.putText(image, class_names[predicted] +" : "+str(logits_numpy[predicted]) , (150, 60), cv.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255))
    cv.imwrite(os.path.join(base_name,base),image)

    mask_pp, _ = gradcam_pp(inputs)
    mask_pp = mask_pp.cpu().detach().numpy()
    heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

    grad_images.append(torch.stack([torch_img.squeeze().cpu(), heatmap,  result], 0))

grad_images = make_grid(torch.cat(grad_images, 0), nrow=3)

output_dir = './1128_result'
os.makedirs(output_dir, exist_ok=True)
output_name = "model_test.jpg"
output_path = os.path.join(output_dir, output_name)

save_image(grad_images, output_path)
Image.open(output_path)


'''
Sample run: python test.py test
'''
