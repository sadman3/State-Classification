
# you can import pretrained models for experimentation & add your own created models
from torchvision.models import resnet18, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch
import torch.nn as nn
import torch.nn.functional as F

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class fc_model(nn.Module):

    def __init__(self, input_size, num_classes=11, dropout=0.5):
        """
            A linear model for image classification.
        """

        super(fc_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.fc1 = nn.Linear(1152, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.adaptivepooling = nn.AdaptiveAvgPool2d(output_size=7)
        self.dropout = nn.Dropout(dropout)
        self.bn_conv1 = nn.BatchNorm2d(16)
        self.bn_conv3 = nn.BatchNorm2d(64)
        self.bn_conv2 = nn.BatchNorm2d(32)
        self.bn_conv3 = nn.BatchNorm2d(64)
        self.bn_conv4 = nn.BatchNorm2d(64)
        self.bn_conv5 = nn.BatchNorm2d(128)
        self.bn_conv6 = nn.BatchNorm2d(128)
        self.bn_fc1 = nn.BatchNorm1d(128)

        

    def forward(self, image_vectorized):
        """
            feed-forward (vectorized) image into a linear model for classification.   
        """

        
        out = self.relu(self.bn_conv1(self.conv1(image_vectorized)))
        out = self.maxpooling(out)
        out = self.relu(self.bn_conv2(self.conv2(out)))
        out = self.maxpooling(out)
        out = self.relu(self.bn_conv3(self.conv3(out)))
        out = self.maxpooling(out)
        out = self.relu(self.bn_conv4(self.conv4(out)))
        out = self.maxpooling(out)
        out = self.relu(self.bn_conv5(self.conv5(out)))
        out = self.maxpooling(out)
        out = self.relu(self.bn_conv6(self.conv6(out)))
        out = self.maxpooling(out)

        out = out.view(-1, 1152)
        out = self.dropout(out)
        out = self.relu(self.bn_fc1(self.fc1(out)))
        out = self.dropout(out)
        out = self.fc2(out)

        
        return out
        
    
# =======================================
