from PIL import Image
import cv2 as cv
import os
import numpy as np
import torch

dir = "./data_new_3"

def preprocessing_dataset(image,filename):
    base = os.path.basename(filename)

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
    im_bi = cv.bilateralFilter(result, 9, 15, 15) #경계선 뚜렷
    result = cv.GaussianBlur(im_bi, (5, 5), 0.75, 0.75) #노이즈 제거

    gray = cv.cvtColor(result, cv.COLOR_BGR2GRAY)
    kernel = np.ones((11, 11), np.uint8)
    opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel, iterations=4)
    edged = cv.Canny(opening, 10, 50)  # 10, 50

    # cv.imshow("edge", edged)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        crop_image = image
    else:
        #draw_image = result.copy()

        boxs = list()
        for cnts in contours:
            x, y, w, h = cv.boundingRect(cnts)
            boxs.append([x, y, w, h])
            #cv.rectangle(draw_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        [x, y, w, h] = max(boxs, key=lambda x: x[2] * x[3])
        #cv.rectangle(draw_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        #cv.imwrite(os.path.join(dir, base), draw_image)



        # crop
        if w > 450 and h > 400:
            crop_image = result[y:y + h, x:x + w]
        else:
            # print(base , w, h)
            # cv.imshow(filename, draw_image)
            # cv.waitKey(0)
            cv.destroyAllWindows()
            crop_image = image

    img = cv.cvtColor(crop_image, cv.COLOR_BGR2RGB)

    im_pil = Image.fromarray(img)
    return im_pil


def preprocessing(image):
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
    edged = cv.Canny(opening, 10, 50)  # 10, 50

    # cv.imshow("edge", edged)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    if len(contours) ==0:
        crop_image = image
    else:
        # draw_image = result.copy()
        boxs = list()
        for cnts in contours:
            x, y, w, h = cv.boundingRect(cnts)
            boxs.append([x, y, w, h])

        [x, y, w, h] = max(boxs, key=lambda x: x[2] * x[3])
        # cv.rectangle(draw_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

        # crop
        if w > 450 and h > 400:
            crop_image = result[y:y + h, x:x + w]
        else:
            crop_image = image

    img = cv.cvtColor(crop_image, cv.COLOR_BGR2RGB)

    #print(img.shape)
    #cv.imshow("crop image", img)
    #cv.waitKey(0)
    #cv.destroyAllWindows()


    im_pil = Image.fromarray(img)
    return im_pil


def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def find_densenet_layer(arch, target_layer_name):
    """Find densenet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_transition1'
            target_layer_name = 'features_transition1_norm'
            target_layer_name = 'features_denseblock2_denselayer12'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'classifier'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer


def find_efficient_layer(arch, target_layer_name):
    """Find efficientnet layer to calculate GradCAM and GradCAM++


    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = target_layer_name.split('.')

    target_layer=arch._modules[hierarchy[0]]._modules['15']

    return target_layer




def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """

    heatmap = cv.applyColorMap(np.uint8(255 * mask.squeeze()), cv.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)


def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def to_one_hot_vector(num_class, label):

   return np.squeeze(np.eye(num_class)[label])

def acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    print(y_pred_tag)
    print(y_test)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    print(correct_results_sum)
    print(y_test.shape[0])
    acc = correct_results_sum / y_test.shape[0]
    acc = torch.round(acc * 100)

    return acc