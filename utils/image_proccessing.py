import torch
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageFont, ImageDraw
from torchvision import transforms

import matplotlib.pyplot as plt
import torch.nn as nn

def add_border(image):
        pad = 10
        mask1 = np.uint8(np.ones((int(image.shape[0] - 2 * pad), int(image.shape[1] - 2 * pad))) * 255.0)
        mask2 = np.pad(mask1, pad_width=pad)
        print(image.shape, mask1.shape, mask2.shape)
        res = cv2.bitwise_and(mask2, image)
        res = cv2.bitwise_or(cv2.bitwise_not(mask2), res)
        return res

def add_contrast(x, factor):
    return transforms.functional.adjust_contrast(x, factor)