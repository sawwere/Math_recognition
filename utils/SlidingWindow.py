import torch
import numpy as np
import sys
import time
import base64
import cv2
import pytesseract

import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFont, ImageDraw
from torchvision import transforms

import imutils
from imutils.object_detection import non_max_suppression

import models.MathNet as mnt
import models.MathNet112 as mnt56
from models.MathNetFactory import MathNetFactory
from utils.letter import Letter
from utils.printer import PrettyPrinter

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"



def intersect_rect(x1, y1, x2, y2, xa, ya, xb, yb):
    xx1 = max(x1, xa)
    xx2 = min(x2, xb)
    yy1 = max(y1, ya)
    yy2 = min(y2, yb)
    return (xx1, yy1, xx2, yy2)

def custom_sort(countour):
        return -cv2.contourArea(countour)

def add_contrast(x, factor):
    return transforms.functional.adjust_contrast(x, factor)

class SlidingWindow():
    def __init__(self, model, kwargs):
        self.model = model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
        self.kwargs = kwargs
        self.avg = 0

    def mean(self, contours):
        if len(contours) <= 0:
            return (0, 0)
        avg_w = 0
        avg_h = 0
        for contour in contours:
            (x, y, w, h) = cv2.boundingRect(contour)
            avg_w += w
            avg_h += h
        avg_w /= len(contours)
        avg_h /= len(contours)
        
        return (int(avg_w), int(avg_h))

    
    def preprocess(self, image, frame):
        res = image.copy()
        gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.blur(gray, (3, 3))
        thresh = 110
        #thresh_img = cv2.threshold(blurred,0,255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)[1]
        thresh_img = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,8)
        contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        mask = np.uint8(np.zeros((thresh_img.shape[0], thresh_img.shape[1])))
        for (idx, contour) in enumerate(contours[1:]):
            (x, y, w, h) = cv2.boundingRect(contour)
            if (x - frame[0]) * (x -  image.shape[1] - frame[0]) > 0 or (y - frame[1]) * (y -  image.shape[0] - frame[1]) > 0 :
                cv2.drawContours(mask, [contour], 0, (255), -1)
                
        # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)   
        # self.kwargs['ROI_SIZE'] = self.mean(contours)
        # print(self.kwargs['ROI_SIZE'])
        result = cv2.bitwise_or(thresh_img, mask)
        #result = self.__add_border(result)
        #result = cv2.erode(result, np.ones((3, 3), np.uint8), iterations=2 )
        if self.kwargs['DEBUG'] == True:
            cv2.imshow('result', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return result
    
    def sliding_window(self, image, window_stride, step):
        for y in range(0, image.shape[0] - window_stride[1], step):
            for x in range(0, image.shape[1] -  window_stride[0], step):
                yield (x, y, image[y:y +  window_stride[1], x:x +  window_stride[0]])
    
    def image_pyramid(self, image, scale, minSize):
        yield image
        while True:
            w = int(image.shape[1] / scale)
            image = imutils.resize(image, width=w)
            if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
                break
            #cv2.imshow("{} {};".format(w, image.shape[0]), image)
            #cv2.waitKey(0)
            yield image   

    def __add_border(self, image):
        pad = 10
        mask1 = np.uint8(np.ones((int(image.shape[0] - 2 * pad), int(image.shape[1] - 2 * pad))) * 255.0)
        mask2 = np.pad(mask1, pad_width=pad)
        print(image.shape, mask1.shape, mask2.shape)
        res = cv2.bitwise_and(mask2, image)
        res = cv2.bitwise_or(cv2.bitwise_not(mask2), res)
        return res

    def get_rois(self, W, processed):
        pyramid = self.image_pyramid(processed, scale=self.kwargs['PYR_SCALE'], minSize=self.kwargs['ROI_SIZE'])
        res = []
        strideX = self.kwargs['ROI_SIZE'][0]
        strideY = self.kwargs['ROI_SIZE'][1]
        for image in pyramid:
            scale = W / float(image.shape[1])
            #cv2.imshow("{} {};".format(scale, image.shape[0]), image)
            #cv2.waitKey(0)
            for (x, y, roiOrig) in self.sliding_window(image, (strideX, strideY), self.kwargs['WIN_STEP']):
                #roiOrig = self.__add_border(roiOrig)
                img = Image.fromarray(roiOrig.astype('uint8'))
                # skip blank images   
                area = img.size[0]*img.size[1]
                black_count = area - cv2.countNonZero(roiOrig)   
                #if ( black_count / area > 0.01):         
                if (img.size[0]*img.size[1] - cv2.countNonZero(roiOrig) != 0):
                    x = int(x * scale)
                    y = int(y * scale)
                    w = int(self.kwargs['ROI_SIZE'][0] * scale)
                    h = int(self.kwargs['ROI_SIZE'][1] * scale)
                    temp_image = processed[y:y+h, x:x+w]
                    _, temp_image = cv2.threshold(temp_image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    roi = cv2.resize(temp_image, self.kwargs['INPUT_SIZE'])
                    #roi = cv2.dilate(roi, np.ones((3, 3), np.uint8), iterations=1)
                    res.append(Letter(x, y, w, h, roi))
        return res

    def visualize_rois(self, img, rois):
        output = Image.fromarray(img.astype('uint8'))        
        res_letters = []
        for idx in range(0, len(rois)):
            roi = rois[idx]
            res_letters.append(roi)
            draw = ImageDraw.Draw(output)
            draw.rectangle((roi.x, roi.y, roi.right, roi.bottom), outline=(0,255,0))
        output.show()
        return (output, res_letters)
        
    def visualize_preds(self, img, letters):
        output = Image.fromarray(img.astype('uint8'))
        font = ImageFont.truetype("T:\my_programs\Math_recognition\ARIALUNI.TTF", 10, encoding="unic")
        draw = ImageDraw.Draw(output)
        res_letters = []
        for letter in letters:
            res_letters.append(letter)
            draw.rectangle((letter.x, letter.y, letter.x+letter.width, letter.y+letter.height), outline=(0,255,0))
            draw.text((letter.x, letter.y), "{}; {:.3f}.".format(mnt.map_pred(letter.value), letter.score), font=font, fill=(200,40,0,255))
        if self.kwargs['DEBUG'] == True:
            output.show()
        return (output, res_letters)

    def get_exact_locations(self, rois):
        res = []
        for letter in rois:
            img = self.__add_border(letter.image)
            img = cv2.bitwise_not(img)
            # Get contours
            contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)    
            if len(contours) < 2:
                continue
            img_contours = np.uint8(np.zeros((img.shape[0],img.shape[1])))
            cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
            # Filter contours
            contours = list(contours)
            contours.sort(key=custom_sort)
            #print(len(contours))
            my_countour = contours[1 if len(contours) > 1 else 0]
            (x, y, w, h) = cv2.boundingRect(my_countour)
            #print(x, y, w, h, img.shape[0]*img.shape[1])
            print(cv2.contourArea(my_countour))
            if cv2.contourArea(my_countour) < 100 or w*h > img.shape[0]*img.shape[1] * 0.8:
                continue
            
            #print("R", x, y, w, h, cv2.contourArea(contour))
            crop_img = img[y:y+h, x:x+w]
            crop_img = cv2.bitwise_not(crop_img)
            cv2.imshow('img', img)
            cv2.imshow('crop', crop_img)
            cv2.waitKey(0)
            #exit(0)
            _let = Letter(x+letter.x, y+letter.y, w, h, crop_img)
            res.append(_let)

        print(len(res))
        return res
    
    def predict(self, letters):
        regions_of_interest = []
        labels = {}
        torch.cuda.empty_cache()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for letter in letters:
            img = Image.fromarray(letter.image.astype('uint8'))
            #print(img.size)
            convert_tensor = transforms.Compose([
                transforms.Resize(self.kwargs['INPUT_SIZE']),
                transforms.Grayscale(1),
                # transforms.ToTensor()

            ])
            x_image = convert_tensor(img)
            x_image = add_contrast(x_image, 5)
            to_tensor = transforms.ToTensor()

            x_image = to_tensor(x_image)
            
            x_image = x_image.unsqueeze(0).float()
            x_image = x_image.to(device)

            predicted = self.model(x_image) 
            #predicted[0][29] = -100
            prob = predicted.max().item()
            
            #print(prob)
            if prob >= self.kwargs['MIN_CONF']:   
                value = predicted.argmax().item()
                letter.value = value
                #letter.value += "; {:.2f}".format(prob)
                letter.score = prob

                ll = labels.get(value, [])
                ll.append(letter)
                labels[value] = ll
                regions_of_interest.append(letter)
                #display(img)
        return (regions_of_interest, labels)
    
    def non_max_suppression(self, preds : dict, thresh : float):
        res = []
        for label in preds.keys():
            letters = preds[label]
            letters.sort(key=lambda ll: (ll.score), reverse=False)
            while len(letters) > 0:
                cur_letter = letters[-1]
                res.append(cur_letter)
                letters = letters[:-1]

                if len(letters) == 0:
                    break
                new_list = []
                for letter in letters:
                    (xx1, yy1, xx2, yy2) = intersect_rect(cur_letter.left, cur_letter.top, cur_letter.right, cur_letter.bottom,
                                         letter.left, letter.top, letter.right, letter.bottom)
                    w = xx2 - xx1
                    h = yy2 - yy1
                    if w < 0 or h < 0:
                        new_list.append(letter)
                        continue
                    intersection = w*h
                    union = cur_letter.width * cur_letter.height + letter.width * letter.height - intersection
                    iou = intersection / union
                    if iou < thresh:
                        new_list.append(letter)
                letters = new_list
        return res

    def __call__(self, img):
        processed_image = self.preprocess(img, (45, 45))
        regions_of_interest = self.get_rois(img.shape[1], processed_image)
        #if self.kwargs['DEBUG'] == True:
        #    self.visualize_rois(img, regions_of_interest)

        if self.kwargs['DEBUG'] == True:
            print('regions_of_interest = ', len(regions_of_interest))

        # regions_of_interest = self.get_exact_locations(regions_of_interest)
        #self.add_spaces_to_letter(regions_of_interest)

        (regions_of_interest, preds) = self.predict(regions_of_interest)
        if self.kwargs['DEBUG'] == True:
            print('letters predicted = ', len(regions_of_interest))
            if self.kwargs['VISUALIZE'] == True:
                self.visualize_preds(img, regions_of_interest)
        regions_of_interest = self.non_max_suppression(preds, self.kwargs['IOU_THRESH'])
        if self.kwargs['DEBUG'] == True:
            print('apply non_max_suppression = ', len(regions_of_interest))

        if self.kwargs['DEBUG'] == True:
            print('found letters = ', len(regions_of_interest))
        (_, letters) = self.visualize_preds(img, regions_of_interest)
        printer = PrettyPrinter()
        printer.print(letters)
        return (_, letters)



if __name__ == '__main__':
    debug = True
    IMAGE_SIZE = int(sys.argv[1])
    path = sys.argv[2]
    if (sys.argc > 3):
        if (sys.argv[3] == "REL"):
            debug = False
    print(IMAGE_SIZE)
    kwargs = dict(
        PYR_SCALE=2.25,
        WIN_STEP=16,
        ROI_SIZE=(48, 48),
        INPUT_SIZE=(IMAGE_SIZE, IMAGE_SIZE),
        VISUALIZE=True,
        MIN_CONF = 6.05,
        IOU_THRESH = 0.5,
        DEBUG=debug
    )
    image = cv2.imread(path)
    factory = MathNetFactory()
    factory.SetModel(IMAGE_SIZE)
    sw = SlidingWindow(factory.LoadModel(), kwargs)
    res = sw(image)
    print(base64.decode(res[0]))
    