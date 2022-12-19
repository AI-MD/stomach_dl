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

class_names=['C','D','E','S1', 'S2', 'S3', 'S4', 'S5','S6']
cnt = 10
f = open("./log.txt", 'w')
def inference(model,images,check_class,input ,save_path, check_cls,threshold):
    all_right = True



    input_size = (int(input), int(input))

    # Preprocessing transformations
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
            if predicted[0][8] >= threshold[5]:
                predicted_classes.append("X")

            if predicted[0][3] >= threshold[0]:
                check_class[class_names[3]] = True
                predicted_classes.append(class_names[3])
            elif predicted[0][4] >= threshold[1]:
                check_class[class_names[4]] = True
                predicted_classes.append(class_names[4])
            elif predicted[0][5] >= threshold[2]:
                check_class[class_names[5]] = True
                predicted_classes.append(class_names[5])
            elif predicted[0][6] >= threshold[3]:
                check_class[class_names[6]] = True
                predicted_classes.append(class_names[6])
            elif predicted[0][7] >= threshold[4]:
                check_class[class_names[7]] = True
                predicted_classes.append(class_names[7])
            else:
                predicted_classes.append("X")

            if check_cls !="all":
                if predicted_classes[0] == check_cls:
                    all_right = False
                    cv.imwrite(os.path.join(save_path, base), image);
            else:
                if label is not predicted_classes[0]:
                    if predicted_classes[0] == "X":
                       cv.imwrite(os.path.join(save_path, base), image);
                       # cv.imwrite(os.path.join(save_path, base), image);

        global cnt

        if all_right is not True:
            cnt = cnt - 1

        print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간
        print(dir, " : ", check_class)


def main():


    IMDIR = sys.argv[1]
    input = sys.argv[2]
    save_path  = sys.argv[3]
    check  = sys.argv[4]
    model = sys.argv[5]

    # cut_off = int(sys.argv[7])
    # if cut_off == 99:
    #     threshold = [
    #         [0.99, 0.99, 0.99, 0.99, 0.99],
    #         [0.99, 0.99, 0.99, 0.99, 0.99],
    #         [0.99, 0.99, 0.99, 0.99, 0.99],
    #         [0.99, 0.99, 0.99, 0.99, 0.99]
    #     ]
    # else:
    #     threshold = [
    #         [0.50, 0.50, 0.50, 0.50, 0.50],
    #         [0.50, 0.50, 0.50, 0.50, 0.50],
    #         [0.50, 0.50, 0.50, 0.50, 0.50],
    #         [0.50, 0.50, 0.50, 0.50, 0.50]
    #     ]

    threshold = [ 0.98, 0.99, 0.97, 0.98, 0.98, 0.92]

    MODEL = model

    model = torch.load(MODEL)
    model.eval()

    folders = os.listdir(IMDIR)
    for folder in folders:
        folder_path = os.path.join(IMDIR, folder)
        files = Path(folder_path).resolve().glob('*.*')
        images = list(files)

        check_class = {
            'S1': False,
            'S2': False,
            'S3': False,
            'S4': False,
            'S5': False
        }

        inference(model, images, check_class,input,save_path,check,threshold)
    print("맞춘 개수", cnt)
    f.close()
if __name__ == "__main__":
	main()












