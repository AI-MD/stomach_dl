
import os
import cv2 as cv
import numpy as np
from PIL import Image
from pathlib import Path

source="E:/stomach/project/pytorch-image-classification/data_new"


files=Path(source).resolve().glob('*.jpg')
filepaths=list(files)

cnt=0
origin_dir = "../data_new/bad_3"
edge_dir = "../data_new/edge_new_3"
conver_dir= "../data_new_2"

for idx,image_path in enumerate(filepaths):
    base = os.path.basename(image_path)
    image= cv.imread(os.path.abspath(image_path))

    resize_image = cv.resize(image, dsize=(640, 480), interpolation=cv.INTER_AREA)

    ht, wd, cc = resize_image.shape

    # create new image of desired size and color (blue) for padding
    ww = wd + 60
    hh = ht + 60
    color = (0, 0, 0)
    result = np.full((hh, ww, cc), color, dtype=np.uint8)

    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    result[yy:yy + ht, xx:xx + wd] = resize_image

    # cv.imshow("padding image", result)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    im_bi = cv.bilateralFilter(result, 9, 15, 15)
    result = cv.GaussianBlur(im_bi, (5, 5), 0.75, 0.75)

    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    kernel = np.ones((11, 11), np.uint8)
    opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel, iterations=4)
    edged = cv.Canny(opening, 10, 50) # 5, 50

    # cv.imshow("edge", edged)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        crop_image = image
    else:
        draw_image = result.copy()

        boxs = list()
        for cnts in contours:
            x, y, w, h = cv.boundingRect(cnts)
            boxs.append([x, y, w, h])
            # cv.rectangle(draw_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        [x, y, w, h] = max(boxs, key=lambda x: x[2] * x[3])
        cv.rectangle(draw_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        cv.imwrite(os.path.join(conver_dir, base), draw_image)

        # cv.imshow(filename, drop_image)
        # cv.waitKey(0)
        # cv.destroyAllWindows()

        # crop
        if w > 470 and h > 400:
            crop_image = result[y:y + h, x:x + w]
        else:
            cv.imshow("draw image", draw_image)
            cv.waitKey(0)
            cv.destroyAllWindows()
            print(base)
            crop_image = image



