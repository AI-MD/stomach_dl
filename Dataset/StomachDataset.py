import torch.utils.data as data
import os
from PIL import Image
import cv2 as cv
import numpy as np
from util.utils import preprocessing_dataset



class StomachDataset(data.Dataset):

    def read_data_set(self):

        all_img_files = []
        all_labels = []

        class_names = os.walk(self.data_set_path).__next__()[1]

        for index, class_name in enumerate(class_names):
            label = index
            img_dir = os.path.join(self.data_set_path, class_name)
            img_files = os.walk(img_dir).__next__()[2]

            for img_file in img_files:
                img_file = os.path.join(img_dir, img_file)
                img = Image.open(img_file)
                if img is not None:
                    all_img_files.append(img_file)
                    all_labels.append(label)

        return all_img_files, all_labels, len(all_img_files), len(class_names),class_names

    def __init__(self, data_set_path, transforms=None):
        self.data_set_path = data_set_path
        self.image_files_path, self.labels, self.length, self.num_classes,self.classes  = self.read_data_set()
        self.transforms = transforms


    def __getitem__(self, index):
        image = cv.imread(self.image_files_path[index])

        im_pil=preprocessing_dataset(image,self.image_files_path[index])

        if self.transforms is not None:
            img = self.transforms(im_pil)

        return img, self.labels[index]

    def __len__(self):
        return self.length



