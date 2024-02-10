import torch
import numpy as np
import sys
import io
import cv2

from PIL import Image, ImageOps, ImageFont, ImageDraw
from torchvision import transforms

from imutils.object_detection import non_max_suppression

import matplotlib.pyplot as plt
import torch.nn as nn

import models.MathNet as mnt
import models.MathNet56 as mnt56
from models.MathNetFactory import MathNetFactory
from SlidingWindow import *

debug = True
IMAGE_SIZE = int(sys.argv[1])
path = sys.argv[2]
if (len(sys.argv) > 3):
    if (sys.argv[3] == "REL"):
        debug = False

kwargs = dict(
        PYR_SCALE=2.25,
        WIN_STEP=16,
        ROI_SIZE=(48, 48),
        INPUT_SIZE=(IMAGE_SIZE, IMAGE_SIZE),
        VISUALIZE=True,
        MIN_CONF=3.05,
        DEBUG=debug
    )
image = cv2.imread(path)
factory = MathNetFactory()
factory.SetModel(IMAGE_SIZE)
sw = SlidingWindow(factory.LoadModel(), kwargs)
res = sw(image)
img_byte_arr = io.BytesIO()
# define quality of saved array
res[0].save(img_byte_arr, format='JPEG', subsampling=0, quality=100)
# converts image array to bytesarray
img_byte_arr = img_byte_arr.getvalue()
if debug == False:
    print(base64.b64encode(img_byte_arr))