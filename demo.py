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

class_names=['C','D','E','S1', 'S2', 'S3', 'S4', 'S5']



def inference(model,images,check_class, fig):
    rows, cols = 9, 8

    input_size = (224, 224)
    # Preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Enable gpu mode, if cuda available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        start = time.time()
        num_idx = 0
        for num, img in enumerate(images):
            image = cv.imread(os.path.abspath(img))

            base = os.path.basename(os.path.abspath(img))
            dir_path = os.path.dirname(os.path.abspath(img))
            dir = os.path.basename(dir_path)
            label = ""

            check = False
            for cls_name in class_names:
                if cls_name in base:
                    label = cls_name
                    check = True
                    break
            if check is not True:
                label = "X"

            img = preprocessing(image)
            inputs = preprocess(img).unsqueeze(0).to(device)
            _, outputs = model(inputs)

            outputs = torch.sigmoid(outputs)  # <--- since you use BCEWithLogitsLoss
            # round up and down to either 1 or 0
            predicted = outputs.cpu().numpy()

            predicted_classes = []

            check_idx = np.where(predicted[0] > 0.5)[0]
            if 0 in check_idx:
                continue
            idx = np.where(predicted[0] > 0.75)[0]
            if len(idx) > 0:
                for id, val in enumerate(idx):
                    predicted_classes.append(class_names[val])
                    if class_names[val] in check_class:
                        check_class[class_names[val]] = True
            else:
                predicted_classes.append("x")

            if label is not predicted_classes[0]:
                if predicted_classes[0] =="x":
                    cv.imwrite('./result_all_75/false/' + base, image);
                else:
                    print("오탐", predicted_classes[0])
                    cv.imwrite('./result_all_75/err/' + base, image);

            # if predicted_classes[0] =="S3":
            #     cv.imwrite('./result_1109_densenet/s3_err/'+base,image);


            # if len(predicted_classes) > 1:
            #     ax = fig.add_subplot(rows, cols, num_idx + 1)
            #     ax.imshow(img)
            #
            #     ax.set_title("lalel : " + label + "\n " + "1 pred class : " + predicted_classes[0] + "\n " + " 2 pred class : " +
            #                  predicted_classes[1])
            #     ax.set_xticks([]), ax.set_yticks([])
            #     num_idx = num_idx + 1
            #
            # else:
            #     ax = fig.add_subplot(rows, cols, num_idx + 1)
            #     ax.imshow(img)
            #
            #     ax.set_title("lalel : " + label + "\n " + "1 pred class : " + predicted_classes[0])
            #     ax.set_xticks([]), ax.set_yticks([])
            #     num_idx = num_idx + 1
            #
            # fig.tight_layout()
        print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        print(dir, " : ", check_class)
        #plt.show()


def main():
    MODEL = "./model_2020_11_09/densenet_sigmoid_224_16_50_8_train.pth"
    model = torch.load(MODEL)
    model.eval()

    IMDIR = sys.argv[1]
    files = Path(IMDIR).resolve().glob('*.*')

    images = list(files)

    check_class = {
                'S1': False,
                'S2': False,
                'S3': False,
                'S4': False,
                'S5': False
            }
    inference(model, images, check_class, fig=False)

    # folders = os.listdir(IMDIR)
    # for folder in folders:
    #     folder_path = os.path.join(IMDIR, folder)
    #     files = Path(folder_path).resolve().glob('*.*')
    #     images = list(files)
    #
    #     # Configure plots
    #     #fig = plt.figure(figsize=(13, 13))
    #
    #     check_class = {
    #         'S1': False,
    #         'S2': False,
    #         'S3': False,
    #         'S4': False,
    #         'S5': False
    #     }
    #
    #     inference(model, images, check_class, fig=False)

if __name__ == "__main__":
	main()












