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

class InitialBlockSTR(nn.Module):
    def __init__ (self,in_channels = 3,out_channels = 16):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, 
                                out_channels,
                                kernel_size = 3,
                                stride = 2, 
                                padding = 1,
                                bias = False)

        # self.prelu = nn.PReLU(16)
        self.relu = nn.ReLU()

        self.batchnorm = nn.BatchNorm2d(out_channels)
  
    def forward(self, x):
        
        x = self.conv(x)
        x = self.batchnorm(x)
        
        x = self.relu(x)
        
        return x




class InitialBlockMP(nn.Module):
    def __init__ (self,in_channels = 3,out_channels = 13):
        super().__init__()


        self.maxpool = nn.MaxPool2d(kernel_size=2, 
                                      stride = 2, 
                                      padding = 0)

        self.conv = nn.Conv2d(in_channels, 
                                out_channels,
                                kernel_size = 3,
                                stride = 2, 
                                padding = 1,
                                bias = False)

        # self.prelu = nn.PReLU(16)
        self.relu = nn.ReLU()

        self.batchnorm = nn.BatchNorm2d(out_channels)
  
    def forward(self, x):
        
        main = self.conv(x)
        main = self.batchnorm(main)
        
        side = self.maxpool(x)
        
        x = torch.cat((main, side), dim=1)
        x = self.relu(x)
        
        return x

