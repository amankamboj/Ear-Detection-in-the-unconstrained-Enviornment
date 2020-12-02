from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from torch.autograd import Variable


class MultiBoxLayer(nn.Module):
    num_classes = 2
    num_anchors = [4,4,4]
    in_planes = [256,1024,1024] # number of channels in each feature map

    def __init__(self):
        super(MultiBoxLayer, self).__init__()

        self.loc_layers = nn.ModuleList() # Holds submodules in a list.
		# ModuleList can be indexed like a regular Python list, but modules it contains are properly registered, and will be visible by all Module methods.
        #print(self.loc_layers )
        self.conf_layers = nn.ModuleList()
        #print(self.conf_layers)
        for i in range(len(self.in_planes)):
        	self.loc_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*4, kernel_size=3, padding=1))
        	self.conf_layers.append(nn.Conv2d(self.in_planes[i], self.num_anchors[i]*2, kernel_size=3, padding=1))
        #print(self.loc_layers)
        #print(self.conf_layers)

    def forward(self, xs):
        '''
        Args:
          xs: (list) of tensor containing intermediate layer outputs.

        Returns:
          loc_preds: (tensor) predicted locations, sized [N,8732,4].
          conf_preds: (tensor) predicted class confidences, sized [N,8732,Number_of_class].
        '''
        y_locs = []
        y_confs = []
        for i,x in enumerate(xs):
            y_loc = self.loc_layers[i](x)
            N = y_loc.size(0) # N : Batch Size
            #print(y_loc.size(1))
            y_loc = y_loc.permute(0,2,3,1).contiguous() # 0 index contains number of images in a batch, 1 contains number of channels, 2 contains height, 3 contains width.
            y_loc = y_loc.view(N,-1,4) 
            #print(y_loc)
            y_locs.append(y_loc)

            y_conf = self.conf_layers[i](x)
            y_conf = y_conf.permute(0,2,3,1).contiguous() # 0 index contains number of images in a batch, 1 contains number of channels, 2 contains height, 3 contains width.
            y_conf = y_conf.view(N,-1,2)
            #print(y_conf)
            y_confs.append(y_conf)
        #print("hello")
        #print(y_locs)
        #print(y_confs)
        loc_preds = torch.cat(y_locs, 1)
        #print(loc_preds)
        conf_preds = torch.cat(y_confs, 1)
        
        return loc_preds, conf_preds
