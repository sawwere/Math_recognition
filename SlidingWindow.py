import torch
import numpy as np
import argparse
import imutils
import time
import cv2

from PIL import Image, ImageOps, ImageFont, ImageDraw
from torchvision import transforms

from imutils.object_detection import non_max_suppression

import matplotlib.pyplot as plt
import torch.nn as nn

import models.MathNet as mnt
import models.MathNet56 as mnt56
from models.MathNetFactory import MathNetFactory

class Letter:
    def __init__(self, x, y, w, h, img):
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.image = img
        
        self.line = 0
        
        self.bottom = self.y + self.height
        self.top = self.y
        self.left = self.x
        self.right = self.x + self.width

        self.value = 'None'
        
    def resize():
        pass

def custom_sort(countour):
        return -cv2.contourArea(countour)

class SlidingWindow():
    def __init__(self, model, kwargs):
        self.model = model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
        self.kwargs = kwargs
    
    def preprocess(self, image):
        res = image.copy()
        gray = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray, (3, 3))
        thresh = 110
        ret, thresh_img = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        thresh_img = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,8)
        cv2.imshow('img_contours', thresh_img )
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return thresh_img
    
    def get_rois(self, image, step, window_stride):
        potential = []
        #find contours
        avg_w = 0
        avg_h = 0
        cnt = 0
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        img_contours = np.uint8(np.zeros((image.shape[0], image.shape[1])))
        cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
        for idx, contour in enumerate(contours):
            x,y,w,h = cv2.boundingRect(contour)
            if cv2.contourArea(contour) > 50:
                avg_w += w
                avg_h += h
                cnt += 1
        avg_w /= cnt
        avg_h /= cnt
        print(avg_h, avg_w)

        step = avg_w

        for y in range(window_stride[1], image.shape[0] - window_stride[1], int(avg_h / 2)):
            for x in range(window_stride[0], image.shape[1] - window_stride[0], int(avg_w / 3)):
                crop_img = image[y:y + int(avg_h), x:x + int(avg_w)]

                #find contours
                contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_contours = np.uint8(np.zeros((crop_img.shape[0],crop_img.shape[1])))
                cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
                
                if (len(contours) > 1):
                    #print(x,y, len(contours))
                    img = Image.fromarray(crop_img.astype('uint8'))
                    #display(img)
                    letter = Letter(x, y, crop_img.shape[1], crop_img.shape[0], crop_img)
                    potential.append(letter)
        return potential
    
    def predict(self, letters):
        res = []
        ind = 0
        #CUDA_LAUNCH_BLOCKING=1
        torch.cuda.empty_cache()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for letter in letters:
            img = Image.fromarray(letter.image.astype('uint8'))
            convert_tensor = transforms.Compose([
                transforms.Resize((self.kwargs['IMAGE_SIZE'],self.kwargs['IMAGE_SIZE'])),
                transforms.Grayscale(1),
                transforms.ToTensor()

            ])
            x_image = convert_tensor(img)
            x_image = x_image.unsqueeze(0).float()
            x_image = x_image.to(device)

            preds = self.model(x_image) 
            prob = preds.max().item()
            #print(prob)
            if prob >= self.kwargs['MIN_CONF']:       
                letter.value = mnt.map_pred(preds.argmax().item())
                res.append(ind)
                #display(img)
            ind += 1
        return res
        
    def visualize_preds(self, img, letters, indices):
        output = Image.fromarray(img.astype('uint8'))
        for ind in indices:
            letter = letters[ind]
            #rect = cv2.rectangle(output, (letter.x, letter.y), (letter.right, letter.bottom), (0, 255,0), 2)

            font = ImageFont.truetype("ARIALUNI.TTF", 24, encoding="unic")
            draw = ImageDraw.Draw(output)
            draw.rectangle((letter.x, letter.y, letter.x+letter.width, letter.y+letter.height))
            draw.text((letter.x, letter.y), str(letter.value), font=font)
            #cv2.putText(output, str(letter.value), (letter.x, letter.y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
       
        #aaa = Image.fromarray(output.astype('uint8'))
        display(output)
        return letters   

    def get_exact_locations(self, rois):
        res = []
        for letter in rois:
            img = letter.image
            output = img.copy()

            # Get contours
            contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)       
            img_contours = np.uint8(np.zeros((img.shape[0],img.shape[1])))
            cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
            # Filter contours
            
            contours = list(contours)
            contours.sort(key=custom_sort)
            my_countour = contours[0]
            if cv2.contourArea(my_countour) < 10 or cv2.contourArea(my_countour) > 5000:
                pass
            (x, y, w, h) = cv2.boundingRect(my_countour)
            #print("R", x, y, w, h, cv2.contourArea(contour))
            crop_img = img[y:y+h, x:x+w]
            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = crop_img
            elif w < h:
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = crop_img
            else:
                letter_square = crop_img
            rect = cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255,0), 2)
            
            _let = Letter(x+letter.x, y+letter.y, w, h, letter_square)
            res.append(_let)

        res.sort(key=lambda ll: (ll.y, ll.x), reverse=False)
        aaa = Image.fromarray(output.astype('uint8'))
        #display(aaa)
        return res

    def __call__(self, img):
        regions_of_interest = self.get_rois(self.preprocess(img), self.kwargs['WIN_STEP'], self.kwargs['ROI_SIZE'])
        print(len(regions_of_interest))
        letters = self.get_exact_locations(regions_of_interest)
        print(len(letters))
        indices = self.predict(letters)
        print(len(indices))
        #nms_labels = self.apply_nms(labels)
        #if self.kwargs['VISUALIZE']:
        self.visualize_preds(img, letters, indices)
        return 1
    
if __name__ == '__main__':
    image = cv2.imread('TEST//5.jpg')
    