import sys
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
import models.MathNet112 as mnt112
from utils.ContoursDetector import ContoursDetector
from utils.letter import Letter
from utils.printer import PrettyPrinter
from utils.Group import Group
from utils.Node import Node

MODEL_PATH = 'models\mathnet224\mathnet8.ml'
printer = PrettyPrinter()

#CUDA_LAUNCH_BLOCKING=1
torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def contour_detection(filename):
    kwargs = dict(
        INPUT_SIZE=(224, 224),
        VISUALIZE=False,
        MIN_CONF=0.05,
        DEBUG=True
    )
    model = mnt.MathNet()
    model.load_state_dict(torch.load(MODEL_PATH))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    image = cv2.imread(filename)
    detector = sw = ContoursDetector(model, kwargs)
    res = sw(image)
    return res

def split_into_lines(letters) -> dict:
    lines = {}
    letters.sort(key=lambda ll: (ll.y), reverse=False)  
    line = 0
    for i in range (1, len(letters)):
        if letters[i].top > letters[i-1].bottom:
            line += 1
        letters[i].line = line
        #print(mnt.map_pred(letters[i].value), letters[i].top, letters[i-1].bottom, line)
    letters.sort(key=lambda ll: (ll.line, ll.x), reverse=False)
    for letter in letters:
        if letter.line not in lines.keys():
            lines[letter.line] = list()
        lines[letter.line].append(letter)
    return lines

def split_line_into_groups(line:list):
    res = []
    prev = 0
    for i in range(1, len(line)):
        if line[i].left > line[i-1].right + 50:
            res.append(line[prev:i])
            prev = i
    res.append(line[prev:len(line)])
    return res

def step_split_lines(prev_step):
    _letters = prev_step[1]
    _hlines = prev_step[2]
    _hlines.sort(key=lambda ll: (ll.y), reverse=True)
    lines = split_into_lines(_letters)
    line_count = max(lines.keys()) + 1
    for line in lines.values():
        #print(type(line))
        print('{' + ' '.join(map(lambda x: printer.char(x.value), line)) +'}')
    print('============')
    return lines

def step_split_groups(prev_step):
    line_count = max(prev_step.keys()) + 1
    for line_idx in prev_step.keys():
        line_groups = split_line_into_groups(prev_step[line_idx])
        prev_step[line_idx] = []
        for g in line_groups:
            prev_step[line_idx].append(Group(g))
    print('total line groups: ', line_count)
    _ = prev_step[1][1]
    _.print()
    return prev_step

def is_item_appr(item, line):
    return item.left >= line.left - 5 and item.right <= line.right + 5

def build_tree(root:Node, current_line, lines, hlines):
    if (current_line < 1):
        return (root, 1)
    
    hline = None
    for _hline in hlines:
        if (is_item_appr(root.value, _hline) 
                and root.value.top - _hline.bottom > 0 
                and root.value.top - _hline.bottom < 55):
            hline = _hline 
            #print('hline.x: ', hline.x)
            break
    if hline is None:
        return (root, 1)
    possible_children = lines[current_line-1]
    print('possible_children on line ', current_line,' : ', len(possible_children))
    for child in possible_children:
        if (is_item_appr(child, hline)
            and _hline.top - child.bottom > 0 
            and _hline.top - child.bottom < 55 ):
            #print(child.letters[0].value)
            tmp = Node(child)
            (tmp, _) = build_tree(tmp, current_line-1, lines, hlines)
            if root.left is None:
                root.left = tmp
            elif root.right is None:
                root.right = tmp
            else:
                print('Too many children found')    
    return (root, current_line+1)

def step_build_tree(prev_step):
    _res = []
    line_count = max(prev_step.keys()) + 1
    current_line = line_count - 1
    while current_line > 0:
        #current_line = current_line - 1
        if current_line < 0:
            break
        line_groups = prev_step[current_line]
        print('line: ', current_line)
        max_k = 1
        for item in line_groups:
            node = Node(item)
            (node, k) = build_tree(node, current_line, prev_step, _hlines)
            max_k = max(k, max_k)
            _res.append(node)
        current_line -= max_k
            
            
        print('=======================',)
    print('res count:', len(_res))
    _res[1].print()
