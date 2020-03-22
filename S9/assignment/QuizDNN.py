import torch
import torchvision
import numpy as np
import pandas as pd
import cv2

from __future__ import print_function        # Import for print statement
import torch                                 # Import pytorch library
import torch.nn as nn                        # Import neural net module from pytorch
import torch.nn.functional as F              # Import functional interface from pytorch
import torch.optim as optim                  # Import optimizer module from pytorch

from torchvision import datasets, transforms # Import datasets and augmentation functionality from vision module within pytorch
from torchsummary import summary             # Import summary with pytorch
from torchviz import make_dot

from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import matplotlib.pyplot as plt
import tensorflow as tf

from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR

# Subclassing nn.Module for neural networks
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #########################################################################################
        # INPUT BLOCK
        #########################################################################################
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            # nn.Dropout(0.05)
        )
        #########################################################################################
        # CONVOLUTION BLOCK 1
        #########################################################################################
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(0.05)
        )
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            # nn.Dropout(0.05)
        )
        #########################################################################################
        # TRANSITION BLOCK 1
        #########################################################################################
        self.pool1 = nn.Sequential(
            nn.MaxPool2d(2, 2) # output_size = 11
        )
        #########################################################################################
        # CONVOLUTION BLOCK 2
        #########################################################################################
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        #########################################################################################
        # TRANSITION BLOCK 2
        #########################################################################################
        self.pool2 = nn.Sequential(
            nn.MaxPool2d(2, 2) # output_size = 11
        )
        #########################################################################################
        # CONVOLUTION BLOCK 3
        #########################################################################################
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        #########################################################################################
        # GAP BLOCK
        #########################################################################################
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        )
        #########################################################################################
        # OUTPUT BLOCK
        #########################################################################################
        self.linear = nn.Sequential(
            nn.Linear(in_features=128, out_features=10),
            # nn.BatchNorm2d(10),
            # nn.ReLU(),
            # nn.Dropout(dropout_value)
        )
        

    def blocks(self, x):

        '''
        x1 = Input
        x2 = Conv(x1)
        x3 = Conv(x1 + x2)

        x4 = MaxPooling(x1 + x2 + x3)

        x5 = Conv(x4)
        x6 = Conv(x4 + x5)
        x7 = Conv(x4 + x5 + x6)

        x8 = MaxPooling(x5 + x6 + x7)

        x9 = Conv(x8)
        x10 = Conv (x8 + x9)
        x11 = Conv (x8 + x9 + x10)

        x12 = GAP(x11)
        x13 = FC(x12)
        '''

        # Input block 
        x1 = self.input_layer(x)  # 32, 101, 101

        # Conv block 1
        x2 = self.convblock1(x1)  # 32, 101, 101
        x3 = self.convblock2(x1 + x2)  # 32, 101, 101

        # Transition block 1
        x4 = self.pool1(x1 + x2 + x3)

        # Conv block 2
        x5 = self.convblock3(x4)           # 32, 101, 101
        x6 = self.convblock4(x4 + x5)       # 32, 101, 101
        x7 = self.convblock5(x4 + x5 + x6)  # 32, 101, 101

        # Transition block 2
        x8 = self.pool1(x5 + x6 + x7)

        # Conv block 3
        x9 = self.convblock6(x8)           # 32, 101, 101
        x10 = self.convblock7(x8 + x9)       # 32, 101, 101
        x11 = self.convblock8(x8 + x9 + x10)  # 32, 101, 101

        x = self.gap(x11)         # 64,  1,  1
        
        x = torch.flatten(x, 1)

        # Predictor
        x = self.linear(x)

        return x # .view(-1, 2)

    def forward(self, x):
        
        x = self.blocks(x)

        return F.log_softmax(x, dim=-1)