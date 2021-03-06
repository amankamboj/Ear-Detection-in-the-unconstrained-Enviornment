from __future__ import print_function

import os
import argparse
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from ssd import SSD300
from utils import progress_bar
from datagen import ListDataset
from multibox_loss import MultiBoxLoss

from torch.autograd import Variable


parser = argparse.ArgumentParser(description='PyTorch SSD Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
args = parser.parse_args()

use_cuda = torch.cuda.is_available() 
best_loss = float('inf')  # best test loss
start_epoch = 0  # start from epoch 0 or last epoch

# Data
print('==> Preparing data..')
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
'''transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=(0.485,), std=(0.229,))])'''

'''trainset = ListDataset(root='/home/biometric/Ranjeet/Data_For_Faster_RCNN/Iris_jpg', list_file='./voc_data/training.txt', train=True, transform=transform)'''
trainset = ListDataset(root='/media/biometric/Data1/Database/Ear_DataSet/Ear_in_wild/Collectionb_all', list_file='//media/biometric/Data1/Database/Ear_DataSet/Ear_in_wild/GT_collectionb_train.txt', train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=20, shuffle=True, num_workers=3)

testset = ListDataset(root='/media/biometric/Data1/Database/Ear_DataSet/Ear_in_wild/Collectionb_all', list_file='//media/biometric/Data1/Database/Ear_DataSet/Ear_in_wild/GT_collectionb_test.txt', train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=20, shuffle=False, num_workers=3)
print("trainset:")
#print(trainset)
# Modl
net = SSD300()
if args.resume:
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/ubear_400_and_Wild_ear_500_images_model.pth')
    net.load_state_dict(checkpoint['net'])
    best_loss = checkpoint['loss']
    start_epoch = checkpoint['epoch']
else:
    # Convert from pretrained VGG model.
    #print("Om")
    net.load_state_dict(torch.load('./model/ssd.pth'))
    print("Ranjeet")

criterion = MultiBoxLoss()
if use_cuda:
    net = torch.nn.DataParallel(net, device_ids=[0])
    net.cuda()
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0

    for batch_idx, (images, loc_targets, conf_targets) in enumerate(trainloader):
        if use_cuda:
            images = images.cuda()
            loc_targets = loc_targets.cuda()
            #print(loc_targets.size())
            conf_targets = conf_targets.cuda()
            #print(conf_targets.size())
        
        images = Variable(images)
        #print(loc_targets.size(), conf_targets.size())  
       
        optimizer.zero_grad()
        loc_preds, conf_preds = net(images)
        #print('hlo ranjeet')
        #print ("Ranjeet")  
        #print(loc_preds.size())
        #print(loc_preds.size(0), conf_preds.size(0))
        loc_targets.resize_(loc_preds.size(0), 8244, 4)
        conf_targets.resize_(loc_preds.size(0), 8244)
        #conf_preds.resize_(loc_preds.size(0), 6400, 2)
        #print(loc_targets.size())
        #print(conf_targets)
        #print(loc_preds.size())
        #print(conf_preds)
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)
        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
        #print (loss)
        loss.backward()
        optimizer.step()
        train_loss += loss.data[0]
        print('%.3f %.3f' % (loss.data[0], train_loss/(batch_idx+1)))

def test(epoch):
    print('\nTest')
    net.eval()
    test_loss = 0
    for batch_idx, (images, loc_targets, conf_targets) in enumerate(testloader):
        if use_cuda:
            images = images.cuda()
            loc_targets = loc_targets.cuda()
            conf_targets = conf_targets.cuda()
        
        images = Variable(images, volatile=True)
        
        loc_preds, conf_preds = net(images)
        loc_targets.resize_(loc_preds.size(0), 8244, 4)
        conf_targets.resize_(loc_preds.size(0),  8244)
        loc_targets = Variable(loc_targets)
        conf_targets = Variable(conf_targets)
        loss = criterion(loc_preds, loc_targets, conf_preds, conf_targets)
        test_loss += loss.data[0]
        print('%.3f %.3f' % (loss.data[0], test_loss/(batch_idx+1)))

    # Save checkpoint.
    global best_loss
    test_loss /= len(testloader)
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.module.state_dict(),
            'loss': test_loss,
            'epoch': epoch,
        }
        states = {'net': net.module.state_dict()}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/Updated_ubear_400_and_Wild_ear_2000_images_model.pth')
        best_loss = test_loss


for epoch in range(start_epoch, start_epoch+1000):
    train(epoch)
    test(epoch)
