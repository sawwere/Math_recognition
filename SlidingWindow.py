import torch
import numpy as np
import sys
import base64
import cv2
import pytesseract

import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageFont, ImageDraw
from torchvision import transforms

import imutils
from imutils.object_detection import non_max_suppression

import models.MathNet as mnt
import models.MathNet56 as mnt56
from models.MathNetFactory import MathNetFactory

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"



class Letter:
    def __init__(self, x, y, w, h, img):
        self.x = x
        self.y = y
        self.width = w
        self.height = h
        self.image = img
        
        self.line = 0

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
        thresh_img = thresh = cv2.threshold(blurred,0,255,cv2.THRESH_OTSU)[1]
        thresh_img = cv2.adaptiveThreshold(blurred, 255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,5,8)
        if self.kwargs['DEBUG'] == True:
            cv2.imshow('thresh_img', thresh_img )
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return thresh_img
    
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

    def get_rois(self, W, pyramid):
        res = []
        for image in pyramid:
            scale = W / float(image.shape[1])
            for (x, y, roiOrig) in self.sliding_window(image, self.kwargs['ROI_SIZE'], self.kwargs['WIN_STEP']):
                if (cv2.countNonZero(roiOrig) > 1):
                    #print(x,y, len(contours))
                    img = Image.fromarray(roiOrig.astype('uint8'))
                    #display(img)
                    x = int(x * scale)
                    y = int(y * scale)
                    w = int(self.kwargs['ROI_SIZE'][0] * scale)
                    h = int(self.kwargs['ROI_SIZE'][1] * scale)
                    
                    roi = cv2.resize(roiOrig, self.kwargs['INPUT_SIZE'])
                    res.append(Letter(x, y, w, h, roi))
                
        return res
    
    def visualize_rois(self, img, letters):
        output = Image.fromarray(img.astype('uint8'))        
        res_letters = []
        for idx in range(0, len(letters)):
            letter = letters[idx]
            res_letters.append(letter)
            draw = ImageDraw.Draw(output)
            draw.rectangle((letter.x, letter.y, letter.x+letter.width, letter.y+letter.height), outline=(255,0,0))
        output.show()
        return (output, res_letters)
        
    def visualize_preds(self, img, letters, indices):
        output = Image.fromarray(img.astype('uint8'))
        res_letters = []
        for ind in indices:
            letter = letters[ind]
            res_letters.append(letter)
            #rect = cv2.rectangle(output, (letter.x, letter.y), (letter.right, letter.bottom), (0, 255,0), 2)
            font = ImageFont.truetype("T:\my_programs\Math_recognition\ARIALUNI.TTF", 14, encoding="unic",)
            draw = ImageDraw.Draw(output)
            draw.rectangle((letter.x, letter.y, letter.x+letter.width, letter.y+letter.height), outline=(255,0,0))
            draw.text((letter.x, letter.y), str(letter.value), font=font, fill=(255,0,0,255))
            #cv2.putText(output, str(letter.value), (letter.x, letter.y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
       
        #aaa = Image.fromarray(output.astype('uint8'))
        if self.kwargs['DEBUG'] == True:
            output.show()
        return (output, res_letters)

    def get_exact_locations(self, rois):
        res = []
        for letter in rois:
            img = letter.image
            # Get contours
            contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)       
            img_contours = np.uint8(np.zeros((img.shape[0],img.shape[1])))
            cv2.drawContours(img_contours, contours, -1, (255,255,255), 1)
            # Filter contours
            
            contours = list(contours)
            contours.sort(key=custom_sort)
            my_countour = contours[0]
            (x, y, w, h) = cv2.boundingRect(my_countour)
            #print(w, h)
            if cv2.contourArea(my_countour) < 10 or cv2.contourArea(my_countour) > 5000:
                pass
            
            #print("R", x, y, w, h, cv2.contourArea(contour))
            crop_img = img[y:y+h, x:x+w]
            
            _let = Letter(x+letter.x, y+letter.y, w, h, crop_img)
            res.append(_let)

        res.sort(key=lambda ll: (ll.y, ll.x), reverse=False)
        return res
    
    def add_spaces_to_letter(self, letters):
        for letter in letters:
            img = letter.image
            (w, h) = (letter.width, letter.height)
            size_max = max(letter.width, letter.height)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:
                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = img
            elif w < h:
                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = img
            else:
                letter_square = img
            
            letter.image = img

    def predict(self, letters):
        res = []
        ind = 0
        torch.cuda.empty_cache()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for letter in letters:
            img = Image.fromarray(letter.image.astype('uint8'))
            #print(img.size)
            convert_tensor = transforms.Compose([
                transforms.Resize(self.kwargs['INPUT_SIZE']),
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
                letter.value += ';' + pytesseract.image_to_string(img, lang="eng")
                res.append(ind)
                #display(img)
            ind += 1
        return res

    def __call__(self, img):
        processed_image = self.preprocess(img)
        pyramid = self.image_pyramid(processed_image, scale=self.kwargs['PYR_SCALE'], minSize=self.kwargs['ROI_SIZE'])
        regions_of_interest = self.get_rois(img.shape[1], pyramid)
        if self.kwargs['DEBUG'] == True:
            self.visualize_rois(img, regions_of_interest)

        if self.kwargs['DEBUG'] == True:
            print('regions_of_interest = ', len(regions_of_interest))

        letters = self.get_exact_locations(regions_of_interest)
        #self.add_spaces_to_letter(letters)

        if self.kwargs['DEBUG'] == True:
            print(len(letters))
        indices = self.predict(letters)
        if self.kwargs['DEBUG'] == True:
            print('found letters = ', len(indices))
        res = self.visualize_preds(img, letters, indices)
        return res



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
        MIN_CONF=3.05,
        DEBUG=debug
    )
    image = cv2.imread(path)
    factory = MathNetFactory()
    factory.SetModel(IMAGE_SIZE)
    sw = SlidingWindow(factory.LoadModel(), kwargs)
    res = sw(image)
    print(base64.decode(res[0]))
    