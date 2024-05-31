import torch
import numpy as np
import cv2

from PIL import Image, ImageOps, ImageFont, ImageDraw
from torchvision import transforms
import torch.nn as nn

import models.MathNet as mnt
from utils.image_processing import *
from utils.letter import Letter
from utils.printer import PrettyPrinter

def average_size(lst):
    if len(lst) <= 0:
        return (0, 0)
    avg_w = 0
    avg_h = 0
    for letter in lst:
        avg_w += letter.width
        avg_h += letter.height
    avg_w /= len(lst)
    avg_h /= len(lst)
    
    return (avg_w, avg_h)

class ContoursDetector():
    def __init__(self, model, kwargs):
        self.model = model
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(device)
        self.model.eval()
        self.kwargs = kwargs
        self.average_size = (1, 1)

    
    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(gray, (3, 3))
        thresh_img = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,13,20)
        thresh_img = add_border(thresh_img)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        img_erode = cv2.erode(closing, np.ones((3, 3), np.uint8), iterations=4)
        
        if self.kwargs['VISUALIZE'] == True:
            cv2.imshow('preprocess', img_erode)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return img_erode

    def get_rois(self, image, orig):
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        blurred = cv2.blur(orig, (3, 3))
        image = add_border(image)
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)       
        img_contours = np.uint8(np.ones((image.shape[0],image.shape[1])))
        # Filter contours
        mask = np.uint8(np.ones((image.shape[0],image.shape[1])) * 255.)
        contours = contours[1:]
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            crop_orig = blurred[y:y+h, x:x+w]
            thresh_img = cv2.adaptiveThreshold(crop_orig, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,31,8)
            black_count = w*h - cv2.countNonZero(thresh_img)
            if black_count > 10:
                mask[y:y+h, x:x+w] = thresh_img
            else:
                pass
        if self.kwargs['VISUALIZE'] == True:
            cv2.imshow('preprocess', mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        mask = cv2.bitwise_not(mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        img_contours = np.uint8(np.zeros((image.shape[0],image.shape[1])))
        cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
        
        letters = []
        for idx, contour in enumerate(contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            if hierarchy[0][idx][3] != -1 or cv2.contourArea(contour) < 15:
                continue
            #_, blurred = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            crop_img = orig[y:y+h, x:x+w]
            letter = Letter(x,y,w,h,crop_img)
            letters.append(letter)
        return letters

    def visualize_rois(self, img, rois):
        output = img.copy()
        output = convert_from_cv2_to_pil(output)       
        res_letters = []
        for idx in range(0, len(rois)):
            roi = rois[idx]
            res_letters.append(roi)
            draw = ImageDraw.Draw(output)
            draw.rectangle((roi.x, roi.y, roi.right, roi.bottom), outline=(255,0,0))
        output.show()
        return (output, res_letters)
        
    def visualize_preds(self, img, letters, hlines):
        #output = Image.new("RGBA", img.size)
        output = convert_from_cv2_to_pil(img)   
        font = ImageFont.truetype("T:\my_programs\Math_recognition\ARIALUNI.TTF", 10, encoding="unic")
        draw = ImageDraw.Draw(output)
        res_letters = []
        for letter in letters:
            res_letters.append(letter)
            draw.rectangle((letter.x, letter.y, letter.x+letter.width, letter.y+letter.height), outline=(0,255,0))
            #draw.text((letter.x, letter.y), "{}; {:.3f}.".format(mnt.map_pred(letter.value), letter.score), font=font, fill=(200,40,0,255))
        for hline in hlines:
            draw.rectangle((hline.x, hline.y, hline.right, hline.bottom), outline=(215,56,0))
        if self.kwargs['VISUALIZE'] == True:
            output.show()
        return (output, res_letters)

    def get_exact_locations(self, rois):
        res = []
        (avg_w, avg_h) = average_size(rois)
        self.average_size = (avg_w, avg_h)
        for letter in rois:
            #remove small noise
            if letter.w < avg_w / 2.75 and letter.height - avg_h / 2.75 < 0:
                continue
            # find horizontal lines
            if (letter.w > avg_w * 1.75):
                letter.value = '_hline'
            _, thresh_img = cv2.threshold(letter.image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            letter.image = thresh_img
            res.append(letter)
        return res
    
    def predict(self, letters):
        regions_of_interest = []
        labels = {}
        hlines = []
        torch.cuda.empty_cache()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for letter in letters:
            if (letter.value[0] == '_'):
                if (letter.value == "_hline"):
                    letter.score = 100
                    hlines.append(letter)
                    continue
                #letter.value = '_line'
            letter.add_frame()            
            img = Image.fromarray(letter.image.astype('uint8'))
            convert_tensor = transforms.Compose([
                transforms.Resize(self.kwargs['INPUT_SIZE']),
                transforms.Grayscale(1),
                transforms.GaussianBlur(3),
                # transforms.ToTensor()
            ])
            x_image = convert_tensor(img)
            x_image = add_contrast(x_image, 5)
            #display(x_image)
            to_tensor = transforms.ToTensor()

            x_tensor = to_tensor(x_image)
            
            x_tensor = x_tensor.unsqueeze(0).float()
            x_tensor = x_tensor.to(device)

            predicted = self.model(x_tensor)
            prob = predicted.max().item()
           
            if prob >= self.kwargs['MIN_CONF']:
                letter.value = predicted.argmax().item()
                letter.score = prob
                ll = labels.get(letter.value, [])
                ll.append(letter)
                labels[letter.value] = ll
                regions_of_interest.append(letter)
                #print(letter.value, prob, printer.char(letter.value))
            else:
                if self.kwargs["DEBUG"] == True:
                    print(prob, mnt.map_pred(predicted.argmax().item()))
        return (regions_of_interest, labels, hlines)
    

    def __call__(self, img):
        processed_image = self.preprocess(img)
        regions_of_interest = self.get_rois(processed_image, img)

        if self.kwargs['DEBUG'] == True:
            print('regions_of_interest = ', len(regions_of_interest))
            #self.visualize_rois(img, regions_of_interest)

        regions_of_interest = self.get_exact_locations(regions_of_interest)
        #self.add_spaces_to_letter(regions_of_interest)

        (regions_of_interest, preds, hlines) = self.predict(regions_of_interest)
        hlines.sort(key=lambda ll: (ll.y), reverse=False)
        if self.kwargs['DEBUG'] == True:
            print('letters predicted = ', len(regions_of_interest))
            print('hlines predicted = ', len(hlines))

        if self.kwargs['DEBUG'] == True:
            print('found letters = ', len(regions_of_interest))
        (_, letters) = self.visualize_preds(img, regions_of_interest, hlines)
        printer = PrettyPrinter()
        printer.print(letters)
        return (_, letters, hlines)