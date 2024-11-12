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


class RDDNeck(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, down_flag, mode,relu, projection_ratio=4):
        
        super().__init__()
        
        # Define class variables
        self.in_channels = in_channels
        
        self.out_channels = out_channels
        self.dilation = dilation
        self.down_flag = down_flag
        self.mode = mode
        self.relu = relu

        if down_flag:
            self.stride = 2
            self.reduced_depth = int(in_channels // projection_ratio)
        else:
            self.stride = 1
            self.reduced_depth = int(out_channels // projection_ratio)

        if self.mode == "maxpool":
            self.maxpool = nn.MaxPool2d(kernel_size = 2,
                                    stride = 2,
                                    padding = 0, return_indices=True)

     
        self.down = nn.Conv2d(in_channels = self.in_channels,
                               out_channels = self.out_channels,
                               kernel_size = 1,
                               stride = 2,
                               padding = 0,
                               bias = False,
                               dilation = 1)

    

        self.conv1 = nn.Conv2d(in_channels = self.in_channels,
                               out_channels = self.reduced_depth,
                               kernel_size = 1,
                               stride = 1,
                               padding = 0,
                               bias = False,
                               dilation = 1)
        
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(in_channels = self.reduced_depth,
                                  out_channels = self.reduced_depth,
                                  kernel_size = 3,
                                  stride = self.stride,
                                  padding = self.dilation,
                                  bias = False,
                                  dilation = self.dilation)
                                  
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(in_channels = self.reduced_depth,
                                  out_channels = self.out_channels,
                                  kernel_size = 1,
                                  stride = 1,
                                  padding = 0,
                                  bias = False,
                                  dilation = 1)
        
        self.relu3 = nn.ReLU()
        
        self.batchnorm1 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm3 = nn.BatchNorm2d(self.out_channels)
        
        
    def forward(self, x):
        
        bs = x.size()[0]
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
                
        # x = self.dropout(x)
        
        # Main Branch
        if self.down_flag:
            if self.mode == "maxpool":
                x_copy, indices = self.maxpool(x_copy)
                if self.in_channels != self.out_channels:
                    out_shape = self.out_channels - self.in_channels
                    extras = torch.zeros((bs, out_shape, x.shape[2], x.shape[3]))
                    if torch.cuda.is_available():
                        extras = extras.cuda()
                    x_copy = torch.cat((x_copy, extras), dim = 1)
            else:
                x_copy = self.down(x_copy)
        
        

        # Sum of main and side branches
        if self.relu == "after":
            x = self.relu3(x) + self.relu3(x_copy)
        else:
            x = x + x_copy
            x = self.relu3(x)
            
        if self.down_flag and self.mode == "maxpool":
            return x, indices
        else:
            return x
