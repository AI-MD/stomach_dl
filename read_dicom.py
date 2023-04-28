import pydicom as dicom
import os
import cv2
import matplotlib.pyplot as plt
import PIL # optional
# make it True if you want in PNG format
# Specify the .dcm folder path
folder_path = "09312881"
# Specify the output jpg/png folder path
jpg_folder_path = "JPG_test"
images_path = os.listdir(folder_path)
for n, image in enumerate(images_path):
    ds = dicom.dcmread(os.path.join(folder_path, image))

    pixel_array_numpy = ds.pixel_array
    im_rgb = cv2.cvtColor(pixel_array_numpy, cv2.COLOR_BGR2RGB)

    image = image.replace('.dcm', '.jpg')

    cv2.imwrite(os.path.join(jpg_folder_path, image), im_rgb)
