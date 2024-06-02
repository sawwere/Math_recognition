import cv2
import numpy
import torch

import models.MathNet as mnt

from utils.Node import Node
from utils.Group import Group
from utils.image_processing import *
from utils.image_info import ImageInfo
from utils.ContoursDetector import ContoursDetector
from utils.printer import PrettyPrinter
from utils.steps.step_result import *

class ImageProcessingStep:
    """Шаг обработки изображений"""

    def process(self,info):
        """Выполнить обработку"""
        pass

class ContourSearchStep(ImageProcessingStep):
    def __init__(self,kwargs, model):
        self.kwargs=kwargs
        self.model = model

    def process(self,info : ImageInfo) -> ContourSearchStepResult:
        """Выполнить обработку"""
        image = info.image
        detector = ContoursDetector(self.model, self.kwargs)
        _res = detector(image)
        result = ContourSearchStepResult(convert_from_pil_to_cv2(_res[0]), _res[1], _res[2])
        return result
    

class GroupSplittingStep(ImageProcessingStep):
    def __init__(self,kwargs):
        self.kwargs=kwargs
        self.printer = PrettyPrinter()

    def __split_into_lines(self, letters) -> dict:
        lines = {}
        letters.sort(key=lambda ll: (ll.y), reverse=False)  
        line = 0
        line_bottom = letters[0].bottom
        for i in range (1, len(letters)):
            if letters[i].top > line_bottom:
                line += 1
                line_bottom = letters[i].bottom
            else:
                line_bottom = max(letters[i-1].bottom, line_bottom)
            letters[i].line = line
            
        letters.sort(key=lambda ll: (ll.line, ll.x), reverse=False)
        for letter in letters:
            if letter.line not in lines.keys():
                lines[letter.line] = list()
            lines[letter.line].append(letter)
        return lines
    
    def __avg_distance_between_characters_on_line(self, line:list) -> float:
        res = 1.0
        for i in range(1, len(line)):
            res += line[i].left - line[i-1].right
        
        return res / len(line)
    
    def __split_line_into_groups(self, line:list):
        res = []
        prev = 0
        for i in range(1, len(line)):
            avg_dist = self.__avg_distance_between_characters_on_line(line) * 4
            if line[i].left > line[i-1].right + avg_dist:
                res.append(line[prev:i])
                prev = i
        res.append(line[prev:len(line)])
        return res

    def process(self, info : ContourSearchStepResult) -> GroupSplittingStepResult:
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
        if self.kwargs['DEBUG'] == True:
            for line in lines.keys():
                for group in lines[line]:
                    print('LINE:', line, ' '.join([mnt.map_pred(x.value) for x in group.letters]))
        result = GroupSplittingStepResult(info.image.copy(), lines, _hlines)
        return result
    
class BuildTreeStep:
    def __init__(self,kwargs):
        self.kwargs=kwargs
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
                root.children.append(tmp)
        return (root, current_line+1)

    def process(self, info : ImageInfo) -> BuildTreeStepResult:
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
        result = BuildTreeStepResult(info.image.copy(), _res)
        return result