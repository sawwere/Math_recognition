import torch
import models
import models.MathNet
import models.alexnet
from utils.steps.image_processing_step import *
from utils.steps.step_result import *
from utils.image_info import ImageInfo
from utils.letter import Letter
from utils.lexer import Lexer

class LexerStep(ImageProcessingStep):
    def __init__(self, kwargs):
        self.kwargs=kwargs

    def process(self,info : GroupSplittingStepResult):
        """Выполнить обработку"""
        if self.kwargs['MODEL_KIND'] == 'ALEX_NET':
            model = models.alexnet.AlexNet(45)
        elif self.kwargs['MODEL_KIND'] == 'RES_NET':
            model = models.MathNet.MathNet(45)
        model.load_state_dict(torch.load(self.kwargs['MODEL_PATH']))
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        for line in info.lines.keys():
            for group in info.lines[line]:
                lexer = Lexer(model=model, input=group)
                lexer.parse()
                _ = ''
                print('LINE:', line, ' '.join([mnt.map_pred(x.value) for x in group.letters]))
                    
        result = LexerStepResult(info.image, info.lines, info.hlines)
        return result