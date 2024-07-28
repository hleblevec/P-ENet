import torch
import torch.nn as nn
import brevitas.nn as qnn
from .common import *
from brevitas.quant import Int8Bias

class UBNeck(nn.Module):
    def __init__(self, in_channels, out_channels, config, mode, shortcut=False, projection_ratio=4, weight_bit_width = 8, act_bit_width = 8):
        
        super().__init__()
        
        # Define class variables
        self.in_channels = in_channels
        self.reduced_depth = int(in_channels / projection_ratio)
        self.out_channels = out_channels
        self.config = config
        self.shortcut = shortcut
        self.mode = mode
        

        if self.mode == "unpool":
            self.up_sc = qnn.QuantConv2d(in_channels = self.in_channels,
                                out_channels = self.out_channels,
                                kernel_size = 1,
                                stride = 1,
                                padding = 0,
                                bias = False,
                                weight_quant = CommonIntWeightPerChannelQuant,
                                weight_bit_width = weight_bit_width)
            self.unpool = nn.MaxUnpool2d(kernel_size = 2,
                                stride = 2)
        elif self.mode == "nearest":
            self.up_sc=torch.nn.Sequential(
            qnn.QuantUpsample(scale_factor=2), 
            qnn.QuantConv2d(in_channels=self.in_channels, 
                            out_channels=self.out_channels, 
                            kernel_size=1, 
                            padding=0, 
                            bias=False,
                            weight_quant = CommonIntWeightPerChannelQuant,
                            weight_bit_width = weight_bit_width))
        elif self.mode == "transposed":
            self.up_sc = qnn.QuantConvTranspose2d(in_channels = self.in_channels,
                                out_channels = self.out_channels,
                                kernel_size = 4,
                                stride = 2,
                                padding = 1,
                                output_padding = 0,
                                weight_quant = CommonIntWeightPerChannelQuant,
                                weight_bit_width = weight_bit_width,
                                bias = False)
        else:
            raise NotImplementedError()

        self.convt1 = qnn.QuantConvTranspose2d(in_channels = self.in_channels,
                                out_channels = self.reduced_depth,
                                kernel_size = 1,
                                padding = 0,
                                bias = False,
                                weight_quant = CommonIntWeightPerChannelQuant,
                                weight_bit_width = weight_bit_width)
        
        
        self.relu1 = qnn.QuantReLU(act_quant =  CommonUintActQuant, bit_width = act_bit_width)
        
        if self.config["upsample"] == "transposed":
            self.up_main = qnn.QuantConvTranspose2d(in_channels = self.reduced_depth,
                                  out_channels = self.reduced_depth,
                                  kernel_size = 4,
                                  stride = 2,
                                  padding = 1,
                                  output_padding = 0,
                                  bias = False,
                                  weight_quant = CommonIntWeightPerChannelQuant,
                                  weight_bit_width = weight_bit_width)
        elif self.config["upsample"] == "nearest":
            self.up_main = qnn.QuantUpsample(scale_factor=2)
        # elif self.config["upsample"] == "nearest":
        #     self.up_main=torch.nn.Sequential(
        #         qnn.QuantUpsample(scale_factor=2), 
        #         qnn.QuantConv2d(in_channels=self.reduced_depth, 
        #                         out_channels=self.reduced_depth, 
        #                         kernel_size=1, 
        #                         padding=0, 
        #                         bias=False,
        #                         weight_quant = CommonIntWeightPerChannelQuant,
        #                         weight_bit_width = weight_bit_width))
        else:
            raise NotImplementedError()
        
        self.relu2 = qnn.QuantReLU(act_quant =  CommonUintActQuant, bit_width = act_bit_width)
        
        self.convt2 = qnn.QuantConvTranspose2d(in_channels = self.reduced_depth,
                                  out_channels = self.out_channels,
                                  kernel_size = 1,
                                  padding = 0,
                                  bias = False)
        
        self.relu3 = qnn.QuantReLU(act_quant =  CommonUintActQuant, bit_width = act_bit_width)
        
        self.batchnorm1 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm3 = nn.BatchNorm2d(self.out_channels)
        
    def forward(self, x, side_input=None):
        x_copy = x
        
        # Main Branch
        x = self.convt1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        
        x = self.up_main(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        
        x = self.convt2(x)
        x = self.batchnorm3(x)
        
        # x = self.dropout(x)
        # Side Branch  
        x_copy = self.up_sc(x_copy)
        if self.mode == "unpool":
            x_copy = self.unpool(x_copy, side_input, output_size=x.size())
        
        # Concat
        if self.shortcut:
            side_input = self.up_sc(side_input)
            if self.config["relu_mode"] == "after":
                x = self.relu3(self.relu3(x) + self.relu3(x_copy)) + self.relu3(side_input)
            else:
                x = x + x_copy + side_input
                x = self.relu3(x)
        else:
            if self.config["relu_mode"] == "after":
                x = self.relu3(x) + self.relu3(x_copy)
            else:
                x = x + x_copy
                x = self.relu3(x)
        return x
