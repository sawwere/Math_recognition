import cv2
import numpy
import torch

import models.MathNet as mnt

from utils.Node import Node
from utils.Group import Group
from utils.image_proccessing import *
from utils.image_info import ImageInfo
from utils.ContoursDetector import ContoursDetector
from utils.printer import PrettyPrinter

class ImageProcessingStep:
    """Шаг обработки изображений"""

    def process(self,info):
        """Выполнить обработку"""
        pass

class ContourSearchStep(ImageProcessingStep):
    def __init__(self,kwargs):
        self.kwargs=kwargs

    def process(self,info : ImageInfo):
        """Выполнить обработку"""
        model = mnt.MathNet()
        model.load_state_dict(torch.load(self.kwargs['MODEL_PATH']))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        image = info.image
        detector = ContoursDetector(model, self.kwargs)
        res = detector(image)
        res_info = ImageInfo(convert_from_pil_to_cv2(res[0]))
        res_info.letters = res[1]
        res_info.hlines = res[2]
        return res_info
    

class GroupSplittingStep(ImageProcessingStep):
    def __init__(self):
        self.printer = PrettyPrinter()

    def __split_into_lines(self, letters) -> dict:
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
    
    def __split_line_into_groups(self, line:list):
        res = []
        prev = 0
        for i in range(1, len(line)):
            if line[i].left > line[i-1].right + 50:
                res.append(line[prev:i])
                prev = i
        res.append(line[prev:len(line)])
        return res

    def process(self, info : ImageInfo):
        _letters = info.letters
        _hlines = info.hlines
        _hlines.sort(key=lambda ll: (ll.y), reverse=True)
        lines = self.__split_into_lines(_letters)
        line_count = max(lines.keys()) + 1
        #for line in lines.values():
            #print(type(line))
            #print('{' + ' '.join(map(lambda x: self.printer.char(x.value), line)) +'}')
        #print('============')
        for line_idx in lines.keys():
            line_groups = self.__split_line_into_groups(lines[line_idx])
            lines[line_idx] = []
            for g in line_groups:
                lines[line_idx].append(Group(g))
        #print('total line groups: ', line_count)
        #_ = lines[1][1]
        #_.print()
        res_info = ImageInfo(info.image.copy())
        res_info.lines = lines
        res_info.hlines = _hlines
        return res_info
    
class BuildTreeStep:
    def __init__(self):
        self.printer = PrettyPrinter()

    def __is_item_appr(self, item, line):
        return item.left >= line.left - 5 and item.right <= line.right + 5

    def __build_tree(self, root:Node, current_line, lines, hlines):
        if (current_line < 1):
            return (root, 1)
        
        hline = None
        for _hline in hlines:
            if (self.__is_item_appr(root.value, _hline) 
                    and root.value.top - _hline.bottom > 0 
                    and root.value.top - _hline.bottom < 55):
                hline = _hline 
                #print('hline.x: ', hline.x)
                break
        if hline is None:
            return (root, 1)
        possible_children = lines[current_line-1]
        #print('possible_children on line ', current_line,' : ', len(possible_children))
        for child in possible_children:
            if (self.__is_item_appr(child, hline)
                and _hline.top - child.bottom > 0 
                and _hline.top - child.bottom < 55 ):
                #print(child.letters[0].value)
                tmp = Node(child)
                (tmp, _) = self.__build_tree(tmp, current_line-1, lines, hlines)
                if root.left is None:
                    root.left = tmp
                elif root.right is None:
                    root.right = tmp
                else:
                    print('ERROR: Too many children found')    
        return (root, current_line+1)

    def process(self, info : ImageInfo):
        _res = []
        line_count = max(info.lines.keys()) + 1
        current_line = line_count - 1
        while current_line > 0:
            #current_line = current_line - 1
            if current_line < 0:
                break
            line_groups = info.lines[current_line]
            #print('line: ', current_line)
            max_k = 1
            for item in line_groups:
                node = Node(item)
                (node, k) = self.__build_tree(node, current_line, info.lines, info.hlines)
                max_k = max(k, max_k)
                _res.append(node)
            current_line -= max_k
            #print('=======================',)
        #print('res count:', len(_res))
        #_res[1].print()
        res_info = ImageInfo(info.image.copy())
        res_info.nodes = _res
        return res_info