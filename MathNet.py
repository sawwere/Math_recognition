import torch
import torch.nn as nn

# classes = ['(', ')', ',', '-', 
#                    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
#                    'A', '[', 'α', 'and', 'β', '∃', 'F', '∀', 
#                    'γ', 'λ', 'μ', 'ω', 'or', 'φ', '→', 
#                    'σ', 'sqrt', 'θ', 'v', 'x', 'y', 'z']

classes = ['(', ')', '+', ',', '-', 
                     '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                     'A', 'F','α','and','β', 'delta', '∃', '∀',
                     'γ', 'λ', 'μ', 'not','ω','or','φ','pi','psi','→',
                     'σ','tau','θ','v','x', 'y', 'z']

NUM_CLASSES = len(classes)


def map_pred(ind):
    if ind < NUM_CLASSES:
        return classes[ind]
    return 'ERROR MAPPIMG'

class ResidiumBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dropout_percentage, is_reducer=True):
        super(ResidiumBlock, self).__init__()
        self.dropout_percentage = dropout_percentage
        self.is_reducer = is_reducer
        if self.is_reducer:
            self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                         kernel_size=3, padding=1, stride=2)
        else:
            self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                         kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.conv2 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                     kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.act1  = torch.nn.ReLU()
        
        self.conv3 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                     kernel_size=3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.conv4 = torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, 
                                     kernel_size=3, padding=1)
        self.bn4 = torch.nn.BatchNorm2d(num_features=out_channels)
        self.act2  = torch.nn.ReLU()
        #self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.dropout = nn.Dropout(p=self.dropout_percentage)

    def forward(self, x):
        identity = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if not self.is_reducer:
            x += identity
        x = self.act1(x)
        
        identity = x
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        
        x = self.dropout(x)
        x += identity
        x = self.act2(x)
        
        return x
    
class MathNet(torch.nn.Module):
    def __init__(self, out_size=NUM_CLASSES):
        super(MathNet, self).__init__()
        self.dropout_percentage = 0.25
        
        # 224x224x1 -> 112x112x64
        self.conv1 =  nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        # 112x112x64 -> 56x56x64
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.block1 = ResidiumBlock(64, 64, self.dropout_percentage, False) # 56x56x64 -> 56x56x64
        self.block2 = ResidiumBlock(64, 128, self.dropout_percentage) # 56x56x64 -> 28x28x128
        self.block3 = ResidiumBlock(128, 256, self.dropout_percentage) # 28x28x128 -> 14x14x256
        self.block4 = ResidiumBlock(256, 512, self.dropout_percentage) # 14x14x256 -> 7x7x512
        # 7x7x512 -> 1x1x512
        self.pool3 = torch.nn.AvgPool2d(kernel_size=7, stride=7, padding=0)
        self.dropout3 = nn.Dropout(p=self.dropout_percentage)
        # 512 -> NUM_CLASSES
        self.fc1 = torch.nn.Linear(512, NUM_CLASSES)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool3(x)

        x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
        x = self.fc1(x)  
        return x