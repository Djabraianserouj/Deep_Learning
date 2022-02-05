import torch 
import torch.nn.functional as F
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, 2, 1)
        self.bn = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, 2)
        self.Resblock1 = ResBlock(64, 64, 1)
        self.Resblock2 = ResBlock(64, 128, 2)
        self.Resblock3 = ResBlock(128, 256, 2)
        self.Resblock4 = ResBlock(256, 512, 2)
        self.globalavgpool = nn.AvgPool2d(10)
        self.FC = nn.Linear(512, 2)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = F.relu(output)
        output = self.maxpool(output)
        output = self.Resblock1(output)
        output = self.Resblock2(output)
        output = self.Resblock3(output)
        output = self.Resblock4(output)
        output = self.globalavgpool(output)
        output = torch.flatten(output, 1)
        output = self.FC(output)
        output = torch.sigmoid(output)

        return output


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, self.stride, 1) #3x3-conv
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1) #3x3-conv
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

        self.conv_1x1 = nn.Conv2d(self.in_channels, self.out_channels, 1, self.stride) #1x1-conv
        self.bn3 = nn.BatchNorm2d(self.out_channels)

    def forward(self, input):
        step = self.conv_1x1(input) #1x1-conv for step 
        step = self.bn3(step)   #batchnorm for step

        output = self.conv1(input)
        output = self.bn1(output)
        output = F.relu(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = output + step      #Add step to the output of BN2
        output = F.relu(output)

        return output 
