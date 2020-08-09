import torch
import torch.nn as nn


# Subclassing nn.Module for neural networks
class Net(nn.Module):
    def __init__(self, load_mobilenet=True):
        super(Net, self).__init__()

        #########################################################################################
        # Mobilenet_v2
        #########################################################################################
        if load_mobilenet:
            self.mobilenet_v2 = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
            # Freezing feature extraction layers
            for param in self.mobilenet_v2.features:
                param.requires_grad = False
            
            # Unfreezing few layers
            for layer_idx in [5, 6, 12, 13, 17, 18]:
                self.mobilenet_v2.features[layer_idx].requires_grad = True

        # _droput = 0.00
        #########################################################################################
        # EXTRA CAPACITY BLOCK
        #########################################################################################

        self.conv_x1_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            # nn.Dropout(_droput)
        )
        
        self.dw_conv_x1_1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            # nn.Dropout(_droput)
        )
        
        self.conv_x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=(64 + 64), out_channels=64, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(in_channels=(64), out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            # nn.Dropout(_droput)
        )
        
        self.dw_conv_x1_2 = nn.Sequential(
            nn.Conv2d(in_channels=(64 + 64), out_channels=64, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            # nn.Dropout(_droput)
        )
        
        self.conv_x2_1 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            # nn.Dropout(_droput)
        )
        
        self.dw_conv_x2_1 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=(1, 1), stride=(2, 2), bias=False),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            # nn.Dropout(_droput)
        )

        self.conv_x3_1 = nn.Sequential(
            nn.Conv2d(in_channels=1280, out_channels=128, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128), 
            nn.ReLU()
            # nn.Dropout(_droput)
        )

        #########################################################################################
        # GAP LAYER
        #########################################################################################
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        #########################################################################################
        # OUTPUT BLOCK
        #########################################################################################
        self.mixer = nn.Sequential(
            nn.Conv2d(in_channels=(128), out_channels=32, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(32)
            # nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=32, out_features=4),
            # nn.Dropout(_droput)
        )

    def blocks(self, x):
        
        x1 = self.mobilenet_v2.features[0:7](x) 
        x2 = self.mobilenet_v2.features[7:14](x1)
        x3 = self.mobilenet_v2.features[14:19](x2)
        
        dw_x1 = self.dw_conv_x1_1(x1)
        x1 = self.conv_x1_1(x1)
        x1 = torch.cat([x1, dw_x1], dim=1)
        
        dw_x1 = self.dw_conv_x1_2(x1)
        x1 = self.conv_x1_2(x1)
        x1 = torch.cat([x1, dw_x1], dim=1)
        
        dw_x2 = self.dw_conv_x2_1(x2)
        x2 = self.conv_x2_1(x2)
        x2 = torch.cat([x2, dw_x2], dim=1)
        
        x3 = self.conv_x3_1(x3)
        x = x1 + x2 + x3
        
        x = self.gap(x)
        x = self.mixer(x)
        x = x.view(-1, 32)

        x1 = self.classifier(x)
        return x1

    def forward(self, x):
        
        x1 = self.blocks(x)
        
        return x1