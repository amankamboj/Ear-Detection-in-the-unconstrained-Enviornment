import math
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from multibox_layer import MultiBoxLayer


class L2Norm2d(nn.Module):
    '''L2Norm layer across all channels.'''
    def __init__(self, scale):
        super(L2Norm2d, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        '''out = scale * x / sqrt(\sum x_i^2)'''
        return self.scale * x * x.pow(2).sum(dim).clamp(min=1e-12).rsqrt().expand_as(x)


class SSD300(nn.Module):
    input_size = 300

    def __init__(self):
        super(SSD300, self).__init__()

        # model
        self.base = self.VGG16()
        
         
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, dilation=1) 
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1) 

# In simple terms, dilated convolution is just a convolution applied to input with defined gaps. 
# With this definitions, given our input is an 2D image, dilation rate k=1 is normal convolution and k=2 means skipping one pixel per input and k=4 means skipping 3 pixels. 
        
        self.norm4 = L2Norm2d(20)
        
        # Before Detection Module
        self.conv4_3_1 = nn.Conv2d(512, 128, kernel_size=1, padding=1, dilation=1)
        self.conv4_3_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, dilation=1)
        
        # Detection Module
        self.conv4_3_3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, dilation=1)
        self.conv4_3_3_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        # Context Module
        self.context1 = nn.Conv2d(128, 64, kernel_size=3, padding=1, dilation=1)
        self.context2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, dilation=1)
        
        self.context1_1 = nn.Conv2d(512,256, kernel_size=3, padding=1, dilation=1)
        self.context2_1 = nn.Conv2d(256,256, kernel_size=3, padding=1, dilation=1)
        
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1)
        self.convd1 = nn.Conv2d(512, 128, kernel_size=1)
	
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, dilation=1)

        self.conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2)

        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=2)

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3)

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3)

        # multibox layer
        self.multibox = MultiBoxLayer()
    def forward(self, x):
        hs = []
        h = self.base(x)
        
        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h_4 = self.norm4(h)
        
        # Before Detection
        
        h = F.max_pool2d(h_4, kernel_size=2, stride=2, ceil_mode=True)
        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h_5 = F.relu(self.conv5_3(h))  
        
        
		# Module 1
        h1d = F.relu(self.conv4_3_1(h_4))
        h2d = F.relu(self.convd1(h_5))
        bi_up = nn.UpsamplingBilinear2d(size= (40,40)) # Bilinear Upsampling
        h_up = bi_up(h2d)
        #print(h_up.size())
        #print(h1d.size())
        element_sum =  torch.add(h1d,h_up)
        
        
        h = F.relu(self.conv4_3_2(element_sum))
        
        # Detection
        h1 = F.relu(self.conv4_3_3(h))
        
        # Context Module
        
        h2 = F.relu(self.context1(h))
        h2_1 = F.relu(self.context2(h2))
        h2_2 = F.relu(self.context2(h2))
        h2_2 = F.relu(self.context2(h2_2))
        
        #print (h2_1.size())
        #print (h2_2.size())
        h_context = torch.cat((h2_1, h2_2), 1)
        #print(h_context.size())
        # Merge in detection module
        h_detection = torch.cat((h1, h_context), 1)
        #print(h_detection.size())
        hs.append(h_detection)  # conv4_3
        
		
		
		
		
		
		# Module 2
		# Detection
        h1 = F.relu(self.conv4_3_3_1(h_5))
        # Context Module
        
        h2 = F.relu(self.context1_1(h_5))
        h2_1 = F.relu(self.context2_1(h2))
        h2_2 = F.relu(self.context2_1(h2))
        h2_2 = F.relu(self.context2_1(h2_2))
        #print (h2_1.size())
        #print (h2_2.size())
        h_context = torch.cat((h2_1, h2_2), 1)
        #print(h_context.size())
        # Merge in detection module
        h_detection = torch.cat((h1, h_context), 1)
        #print(h_detection.size())
        hs.append(h_detection)
		
		
		
		
		
		# Module 3
		# Detection
        h5_max = F.max_pool2d(h_5, kernel_size=2, stride=2, ceil_mode=True)
        
        h1 = F.relu(self.conv4_3_3_1(h5_max))
        # Context Module
        
        h2 = F.relu(self.context1_1(h5_max))
        h2_1 = F.relu(self.context2_1(h2))
        h2_2 = F.relu(self.context2_1(h2))
        h2_2 = F.relu(self.context2_1(h2_2))
        
        #print (h2_1.size())
        #print (h2_2.size())
        h_context = torch.cat((h2_1, h2_2), 1)
        #print(h_context.size())
        # Merge in detection module
        h_detection = torch.cat((h1, h_context), 1)
        #print(h_detection.size())
        hs.append(h_detection)
        #print("hello")
		
        loc_preds, conf_preds = self.multibox(hs)
       
        return loc_preds, conf_preds

    def VGG16(self):
        '''VGG16 layers.'''
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M']
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
            	layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)] # M is used for pooling, 
            																		# ceil_mode: when True, will use `ceil` instead of `floor` to compute the output shape
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(True)] # Here x represents number of filters
                in_channels = x
        return nn.Sequential(*layers)
