import torch
import numpy as np
import imutils
import time
import random
import cv2

from PIL import Image, ImageOps, ImageFont, ImageDraw
from torchvision import transforms

import matplotlib.pyplot as plt
import torch.nn as nn

import models.MathNet as mnt
from utils.image_processing import *
from utils.letter import Letter
from utils.printer import PrettyPrinter

open = ['(']
close = [')']
af = ['A', 'F']
_not = ['not']
implicit = ['I-']
greek = ['alpha','beta','delta','gamma','lambda','mu','omega','phi',
 'pi','psi','sigma','tau','theta','upsilon']
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
logical = ['and', 'or', 'rightarrow', 'equal']

transition_table = {
    'new_line' : af + greek + ['not', '('],
    '(': ['(', 'not', ] + greek,
    ')': [')', 'I-'] + logical,
    'digit': ['(', 'not', '-'].__add__(greek) + digits,
    'AF': digits + ['-'],
    'logical': ['(', '(', 'not', ] + greek,
    'not': ['(', 'not', ] + greek,
    'greek': [')', 'I-'] + logical,
    'I-': ['(', 'not', ] + greek,
    '-' : ['-', '(', 'not'] + greek
    }
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Lexer:
    def __init__(self, model, input, multiplier = 2):
        self.model = model
        self.letters = input.letters
        self.currentChar = None
        self.position = 0
        self.line = 0
        self.lexKind = 'new_line'
        self.mask = torch.from_numpy(np.ones(len(mnt.classes))).to(device)
        self.mask[2] = -1.0

        self.multiplier = multiplier

    def predict(self, char : Letter):
        img = Image.fromarray(char.image.astype('uint8'))
        convert_tensor = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.Grayscale(1),
            ])
        x_image = convert_tensor(img)
        x_image = add_contrast(x_image, 5)
        to_tensor = transforms.ToTensor()

        x_image = to_tensor(x_image)
        
        x_image = x_image.unsqueeze(0).float()
        x_image = x_image.to(device)
        return  self.mask * self.model(x_image)

    def nextChar(self):
        
        if self.position == len(self.letters):
            self.currentChar = None
        else:
            self.currentChar = self.letters[self.position]
            if (self.currentChar.line > self.line):
                self.mask = torch.from_numpy(np.ones(len(mnt.classes))).to(device)
                self.line += 1
            self.position += 1
            

    def nextLexem(self):
        predicted = self.predict(self.currentChar)
        self.currentChar.value = predicted.argmax().item()
        self.currentChar.score = predicted.max().item()
        self.currentChar.char = mnt.map_pred(self.currentChar.value)
        self.mask = torch.from_numpy(np.ones(len(mnt.classes))).to(device)
        char = mnt.map_pred(self.currentChar.value)
        if (self.currentChar is None):
            return None
        if (char == 'A'):
            self.lexKind = 'AF'
        elif (char == 'F'):
            self.lexKind = 'AF'
        elif (char == '('):
            self.lexKind = '('   
        elif (char == ')'):
            self.lexKind = ')'
        elif (char == 'not'):
            self.lexKind = 'not'
        elif (char == '-'):
            self.lexKind = '-'
        elif (char in greek):
            self.lexKind = 'greek'
        elif (char in logical):
            self.lexKind = 'logical'
        elif (char in digits):
            self.lexKind = 'digit'
        else:
            self.lexKind = 'new_line'
        for item in transition_table[self.lexKind]:
                self.mask[mnt.classes.index(item)] *= self.multiplier  
        self.nextChar()    
            
    def parse(self):
        self.nextChar()
        while self.currentChar is not None:
            self.nextLexem()