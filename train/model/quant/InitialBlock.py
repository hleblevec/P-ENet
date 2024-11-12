###################################################
# Copyright (c) 2019                              #
# Authors: @iArunava <iarunavaofficial@gmail.com> #
#          @AvivSham <mista2311@gmail.com>        #
#                                                 #
# License: BSD License 3.0                        #
#                                                 #
# The Code in this file is distributed for free   #
# usage and modification with proper linkage back #
# to this repository.                             #
###################################################

import torch
import torch.nn as nn
import brevitas.nn as qnn
from .common import *

class InitialBlockSTR(nn.Module):
    def __init__ (self, in_channels = 3,out_channels = 16, weight_bit_width = 8, act_bit_width = 8):
        super().__init__()

        self.conv = qnn.QuantConv2d(in_channels, 
                                out_channels,
                                kernel_size = 3,
                                stride = 2, 
                                padding = 1,
                                bias = False,
                                weight_quant = CommonIntWeightPerChannelQuant,
                                weight_bit_width = weight_bit_width)
                                
        self.relu = qnn.QuantReLU(act_quant = CommonUintActQuant, bit_width = act_bit_width)

        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        
        x = self.relu(x)
        
        return x


class InitialBlockMP(nn.Module):
    def __init__ (self, in_channels = 3,out_channels = 13, weight_bit_width = 8, act_bit_width = 8):
        super().__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=2, 
                                      stride = 2, 
                                      padding = 0)

        self.conv = qnn.QuantConv2d(in_channels, 
                                out_channels,
                                kernel_size = 3,
                                stride = 2, 
                                padding = 1,
                                bias = False,
                                weight_quant = CommonIntWeightPerChannelQuant,
                                weight_bit_width = weight_bit_width)

        self.relu = qnn.QuantReLU(act_quant = CommonUintActQuant, bit_width = act_bit_width)

        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        main = self.conv(x)
        main = self.batchnorm(main)

        side = self.maxpool(x)
        
        x = torch.cat((main, side), dim=1)
        x = self.relu(x)
        
        return x
