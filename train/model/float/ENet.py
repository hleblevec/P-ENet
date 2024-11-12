##################################################################
# Reproducing the paper                                          #
# ENet - Real Time Semantic Segmentation                         #
# Paper: https://arxiv.org/pdf/1606.02147.pdf                    #
#                                                                #
# Copyright (c) 2019                                             #
# Authors: @iArunava <iarunavaofficial@gmail.com>                #
#          @AvivSham <mista2311@gmail.com>                       #
#                                                                #
# License: BSD License 3.0                                       #
#                                                                #
# The Code in this file is distributed for free                  #
# usage and modification with proper credits                     #
# directing back to this repository.                             #
##################################################################

import torch
import torch.nn as nn
from .InitialBlock import *
from .RDDNeck import RDDNeck
from .UBNeck import UBNeck
from .ASNeck import ASNeck

class ENet(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Define class variables
        self.config = config
        self.C = config["num_classes"]
        self.ratio=config["projection_ratio"]
        
        # The initial block
        if self.config["Init"] == "MP":
            self.init = InitialBlockMP()
        elif self.config["Init"] == "STR":
            self.init = InitialBlockSTR()
        else:
            raise NotImplementedError()

       
        
        
        # The first bottleneck
        self.b10 = RDDNeck(dilation=1, 
                            in_channels=16, 
                            out_channels=64, 
                            down_flag=True, 
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"])
        
        self.b11 = RDDNeck(dilation=1, 
                            in_channels=64, 
                            out_channels=64, 
                            down_flag=False, 
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"])
        
        self.b12 = RDDNeck(dilation=1, 
                            in_channels=64, 
                            out_channels=64, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"])
        
        self.b13 = RDDNeck(dilation=1, 
                            in_channels=64, 
                            out_channels=64, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"])
        
        self.b14 = RDDNeck(dilation=1, 
                            in_channels=64, 
                            out_channels=64, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"])
        
        
        # The second bottleneck
        self.b20 = RDDNeck(dilation=1, 
                            in_channels=64, 
                            out_channels=128, 
                            down_flag=True,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"])
        
        self.b21 = RDDNeck(dilation=1, 
                            in_channels=128, 
                            out_channels=128,                             
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"], 
                            projection_ratio=self.ratio)
        
        self.b22 = RDDNeck(dilation=2, 
                            in_channels=128, 
                            out_channels=128, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"], 
                            projection_ratio=self.ratio)
        
        self.b23 = ASNeck(in_channels=128, 
                            out_channels=128,
                            config = self.config,
                            projection_ratio=self.ratio)
        
        self.b24 = RDDNeck(dilation=4, 
                            in_channels=128, 
                            out_channels=128, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"], 
                            projection_ratio=self.ratio)
        
        self.b25 = RDDNeck(dilation=1, 
                            in_channels=128, 
                            out_channels=128, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"], 
                            projection_ratio=self.ratio)
        
        self.b26 = RDDNeck(dilation=8, 
                            in_channels=128, 
                            out_channels=128, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"], 
                            projection_ratio=self.ratio)
        
        self.b27 = ASNeck(in_channels=128, 
                            out_channels=128,
                            config = self.config,
                            projection_ratio=self.ratio)
        
        self.b28 = RDDNeck(dilation=16, 
                            in_channels=128, 
                            out_channels=128, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"], 
                            projection_ratio=self.ratio)
        
        
        # The third bottleneck
        self.b31 = RDDNeck(dilation=1, 
                            in_channels=128, 
                            out_channels=128, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"], 
                            projection_ratio=self.ratio)
        
        self.b32 = RDDNeck(dilation=2, 
                            in_channels=128, 
                            out_channels=128, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"], 
                            projection_ratio=self.ratio)
        
        self.b33 = ASNeck(in_channels=128, 
                            out_channels=128, 
                            config = self.config,
                            projection_ratio=self.ratio)
        
        self.b34 = RDDNeck(dilation=4, 
                            in_channels=128, 
                            out_channels=128, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"], 
                            projection_ratio=self.ratio)
        
        self.b35 = RDDNeck(dilation=1, 
                            in_channels=128, 
                            out_channels=128, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"], 
                            projection_ratio=self.ratio)
        
        self.b36 = RDDNeck(dilation=8, 
                            in_channels=128, 
                            out_channels=128, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"], 
                            projection_ratio=self.ratio)
        
        self.b37 = ASNeck(in_channels=128, 
                            out_channels=128,
                            config = self.config,
                            projection_ratio=self.ratio)
        
        self.b38 = RDDNeck(dilation=16, 
                            in_channels=128, 
                            out_channels=128, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"], 
                            projection_ratio=self.ratio)
        
        
        # The fourth bottleneck
        self.b40 = UBNeck(in_channels=128, 
                            out_channels=64,  
                            config=self.config,
                            mode = self.config["UBNeck_mode"],
                            shortcut=self.config["sc2"],
                            projection_ratio=self.ratio)
        
        self.b41 = RDDNeck(dilation=1, 
                            in_channels=64, 
                            out_channels=64, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"])
        
        self.b42 = RDDNeck(dilation=1, 
                            in_channels=64, 
                            out_channels=64, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"],
                            relu = self.config["relu_mode"])
        
        
        # The fifth bottleneck
        self.b50 = UBNeck(in_channels=64, 
                            out_channels=16,
                            config=self.config,
                            mode = self.config["UBNeck_mode"],
                            shortcut=self.config["sc1"])
        
        self.b51 = RDDNeck(dilation=1, 
                            in_channels=16, 
                            out_channels=16, 
                            down_flag=False,
                            mode = self.config["RDDNeck_mode"], 
                            relu = self.config["relu_mode"])
        
        
        # Final ConvTranspose Layer
        if config["upsample"]=="transposed":
            self.last = nn.ConvTranspose2d(in_channels=16, 
                                           out_channels=self.C, 
                                           kernel_size=4, 
                                           stride=2, 
                                           padding=1, 
                                           output_padding=0,
                                           bias=False)
        elif config["upsample"]=="nearest":
            self.last=torch.nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.Conv2d(in_channels=16, 
                        out_channels=self.C, 
                        kernel_size=1, 
                        padding=0, 
                        bias=False))
        
        if self.config["relu_mode"] == "after":
            self.relu=nn.ReLU()    
    
    def forward(self, x):
        
        # The initial block
        x = self.init(x)
        
        # The first bottleneck
        if self.config["RDDNeck_mode"] == "maxpool":
            x, x1 = self.b10(x)
            x = self.b11(x)
        else:
            x1 = self.b10(x)
            x = self.b11(x1)
        x = self.b12(x)
        x = self.b13(x)
        x = self.b14(x)
        
        # The second bottleneck
        if self.config["RDDNeck_mode"] == "maxpool":
            x, x2 = self.b20(x)
            x = self.b21(x)
        else:
            x2 = self.b20(x)
            x = self.b21(x2)
        x = self.b22(x)
        x = self.b23(x)
        x = self.b24(x)
        x = self.b25(x)
        x = self.b26(x)
        x = self.b27(x)
        x = self.b28(x)
        
        # The third bottleneck
        x = self.b31(x)
        x = self.b32(x)
        x = self.b33(x)
        x = self.b34(x)
        x = self.b35(x)
        x = self.b36(x)
        x = self.b37(x)
        x = self.b38(x)
        
        # The fourth bottleneck
        x = self.b40(x,x2)
        x = self.b41(x)
        x = self.b42(x)
        
        # The fifth bottleneck
        x = self.b50(x,x1)
        x = self.b51(x)
        # Final ConvTranspose Layer
        x = self.last(x)
        if self.config["relu_mode"] == "after":
            x = self.relu(x)
        
        return x
