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

class ASNeck(nn.Module):
    def __init__(self, in_channels, out_channels, config, projection_ratio=4, weight_bit_width = 8, act_bit_width = 8):
        
        super().__init__()
        
        # Define class variables
        self.in_channels = in_channels
        self.reduced_depth = int(in_channels / projection_ratio)
        self.out_channels = out_channels
        self.config = config

        
        self.conv1 = qnn.QuantConv2d(in_channels = self.in_channels,
                               out_channels = self.reduced_depth,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               bias = False,
                               weight_quant = CommonIntWeightPerChannelQuant,
                               weight_bit_width = weight_bit_width)
        
        self.relu1 = qnn.QuantReLU(act_quant = CommonUintActQuant, bit_width = act_bit_width)

        if self.config["ASNeck"] == "asym":
            self.conv2=torch.nn.Sequential(
                qnn.QuantConv2d(in_channels = self.reduced_depth,
                                out_channels = self.reduced_depth,
                                kernel_size = (1, 5),
                                stride = 1,
                                padding = (0, 2),
                                bias = False,
                                weight_quant = CommonIntWeightPerChannelQuant,
                                weight_bit_width = weight_bit_width),
                qnn.QuantConv2d(in_channels = self.reduced_depth,
                                out_channels = self.reduced_depth,
                                kernel_size = (5, 1),
                                stride = 1,
                                padding = (2, 0),
                                bias = False,
                                weight_quant = CommonIntWeightPerChannelQuant,
                                weight_bit_width = weight_bit_width))
        elif self.config["ASNeck"] == "asym_relu":
            self.conv2=torch.nn.Sequential(
                qnn.QuantConv2d(in_channels = self.reduced_depth,
                                out_channels = self.reduced_depth,
                                kernel_size = (1, 5),
                                stride = 1,
                                padding = (0, 2),
                                bias = False,
                                weight_quant = CommonIntWeightPerChannelQuant,
                                weight_bit_width = weight_bit_width),
                qnn.QuantReLU(act_quant = CommonUintActQuant, 
                              bit_width = act_bit_width), 
                qnn.QuantConv2d(in_channels = self.reduced_depth,
                                out_channels = self.reduced_depth,
                                kernel_size = (5, 1),
                                stride = 1,
                                padding = (2, 0),
                                bias = False,
                                weight_quant = CommonIntWeightPerChannelQuant,
                                weight_bit_width = weight_bit_width))
        
        elif self.config["ASNeck"] == "2groups":
            self.conv2 = qnn.QuantConv2d(in_channels = self.reduced_depth,
                                    out_channels = self.reduced_depth,
                                    kernel_size = 5,
                                    stride = 1,
                                    padding = 2,
                                    bias = False,
                                    groups = 2,
                                    weight_quant = CommonIntWeightPerChannelQuant,
                                    weight_bit_width = weight_bit_width)
            
        elif self.config["ASNeck"] == "4groups":
            self.conv2 = qnn.QuantConv2d(in_channels = self.reduced_depth,
                                    out_channels = self.reduced_depth,
                                    kernel_size = 5,
                                    stride = 1,
                                    padding = 2,
                                    bias = False,
                                    groups = 4,
                                    weight_quant = CommonIntWeightPerChannelQuant,
                                    weight_bit_width = weight_bit_width)
            
        else:
            raise NotImplementedError()


        self.relu2 = qnn.QuantReLU(act_quant = CommonUintActQuant, bit_width = act_bit_width)
        
        self.conv3 = qnn.QuantConv2d(in_channels = self.reduced_depth,
                                  out_channels = self.out_channels,
                                  kernel_size = 1,
                                  stride = 1,
                                  padding = 0,
                                  bias = False,
                                  weight_quant = CommonIntWeightPerChannelQuant,
                                  weight_bit_width = weight_bit_width)
        
        self.relu3 = qnn.QuantReLU(act_quant = CommonUintActQuant, bit_width = act_bit_width)
        
        self.batchnorm1 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm3 = nn.BatchNorm2d(self.out_channels)

      
        
    def forward(self, x):
        x_copy = x
        
        # Side Branch
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)
        
        # Sum of main and side branches
        if self.config["relu_mode"] == "after":
            x = self.relu3(x) + self.relu3(x_copy)
        else:
            x = x + x_copy
            x = self.relu3(x)
        
        return x