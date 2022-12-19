
import os
import cv2 as cv
import numpy as np
from PIL import Image
from pathlib import Path

source="E:/stomach/project/pytorch-image-classification/after_new_2"

all_img_files = []

f=open("../error_stomach.txt", "w")

files=Path(source).resolve().glob('*.jpg')
filepaths=list(files)

for idx,image_path in enumerate(filepaths):
    image = cv.imread(os.path.abspath(image_path))
    image = cv.resize(image, dsize=(640, 480), interpolation=cv.INTER_AREA)

    ht, wd, cc = image.shape

    # create new image of desired size and color (blue) for padding
    ww = wd + 60
    hh = ht + 60
    color = (0, 0, 0)
    result = np.full((hh, ww, cc), color, dtype=np.uint8)

    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    result[yy:yy + ht, xx:xx + wd] = image

    # cv.imshow("padding image", result)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    # blur 생략
    # result = cv.GaussianBlur(result, (5, 5), 0)

    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    kernel = np.ones((11,11), np.uint8)
    opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel, iterations=4)
    edged = cv.Canny(opening, 50, 200, 10)  # 50,200

    # cv.imshow("edge", edged)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)



    if len(contours) ==0:
        # cv.imshow("edge", edged)
        # cv.waitKey(0)
        # cv.destroyAllWindows()
        # f.write(base)
        # f.write("\n")
        crop_image = image
    else:
        cmax = max(contours, key=cv.contourArea)

        # draw
        drop_image = result.copy()

        x, y, w, h = cv.boundingRect(cmax)
        x2 = x + w
        y2 = y + h
        cv.rectangle(drop_image, (x, y), (x2, y2), (255, 255, 0), 5)

        dir = "../data_new"
        base = os.path.basename(image_path)
        cv.imwrite(os.path.join(dir, base), image)

        angle= (y2-y)/(x2-x)
        #crop
        print(w, h, angle, x, x2, y, y2, base)
        cv.imshow("drop image", drop_image)
        cv.waitKey(0)
        cv.destroyAllWindows()


        if w > 470 and h > 400 and angle > 0.74 and angle < 1.25:
            crop_image = result[y:y + h, x:x + w]
        else:
            # f.write(base)
            # f.write("\n")
            crop_image = image




