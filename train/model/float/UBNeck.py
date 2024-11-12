import torch
import torch.nn as nn

class UBNeck(nn.Module):
    def __init__(self, in_channels, out_channels, config, mode, shortcut=False, projection_ratio=4):
        
        super().__init__()
        
        # Define class variables
        self.in_channels = in_channels
        self.reduced_depth = int(in_channels / projection_ratio)
        self.out_channels = out_channels
        self.config = config
        self.shortcut = shortcut
        self.mode = mode
        

        if self.mode == "unpool":
            self.up_sc = nn.Conv2d(in_channels = self.in_channels,
                                out_channels = self.out_channels,
                                kernel_size = 1,
                                stride = 1,
                                padding = 0,
                                bias = False)
            self.unpool = nn.MaxUnpool2d(kernel_size = 2,
                                stride = 2)
        elif self.mode == "nearest":
            self.up_sc=torch.nn.Sequential(
            nn.Upsample(scale_factor=2), 
            nn.Conv2d(in_channels=self.in_channels, 
                            out_channels=self.out_channels, 
                            kernel_size=1, 
                            padding=0, 
                            bias=False))
        elif self.mode == "transposed":
            self.up_sc = nn.ConvTranspose2d(in_channels = self.in_channels,
                                out_channels = self.out_channels,
                                kernel_size = 4,
                                stride = 2,
                                padding = 1,
                                output_padding = 0,
                                bias = False)
        else:
            raise NotImplementedError()
        
        # self.dropout = nn.Dropout2d(p=0.1)
      
        self.convt1 = nn.ConvTranspose2d(in_channels = self.in_channels,
                                out_channels = self.reduced_depth,
                                kernel_size = 1,
                                padding = 0,
                                bias = False)      
        self.relu1 = nn.ReLU()
        
        if self.config["upsample"] == "transposed":
            self.up_main = nn.ConvTranspose2d(in_channels = self.reduced_depth,
                                  out_channels = self.reduced_depth,
                                  kernel_size = 4,
                                  stride = 2,
                                  padding = 1,
                                  output_padding = 0,
                                  bias = False)
        elif self.config["upsample"] == "nearest":
            self.up_main = nn.Upsample(scale_factor=2)
        else:
            raise NotImplementedError()
        
        self.relu2 = nn.ReLU()

       
        self.convt2 = nn.ConvTranspose2d(in_channels = self.reduced_depth,
                                  out_channels = self.out_channels,
                                  kernel_size = 1,
                                  padding = 0,
                                  bias = False)
       
        self.relu3 = nn.ReLU()
        
        self.batchnorm1 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm2 = nn.BatchNorm2d(self.reduced_depth)
        self.batchnorm3 = nn.BatchNorm2d(self.out_channels)
   
    def forward(self, x, side_input=None):
        x_copy = x
        
        # Side Branch
        x = self.convt1(x)
        x = self.batchnorm1(x)
        x = self.relu1(x)
        
        x = self.up_main(x)
        x = self.batchnorm2(x)
        x = self.relu2(x)
        
        x = self.convt2(x)
        x = self.batchnorm3(x)
        
        # x = self.dropout(x)
        
        # Main Branch

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
