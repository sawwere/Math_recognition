import torch

import models.MathNet as mnt
import models.MathNet20 as mnt20
import models.MathNet112 as mnt112

PREFIX_56 = "models/mathnet56/"
PREFIX_20 = "models/mathnet224/"
PREFIX_224 = "models/mathnet224/"

class MathNetFactory:
    def __init__(self):
        self.model = None
        self.path = ""

    def SetModel(self, image_size):
        if (image_size == 112):
            self.model = mnt112.MathNet112()
        elif (image_size == 20):
            self.model = mnt20.MathNet20()
        elif (image_size == 224):
            self.model = mnt.MathNet()
        else:
            raise Exception('INVALID SIZE')
        self.path = 'T:\my_programs\Math_recognition\models\mathnet'+str(image_size)+'\mathnet.ml'
        


    def LoadModel(self):
        self.model.load_state_dict(torch.load(self.path))
        return self.model