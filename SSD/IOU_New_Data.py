import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
from torch.autograd import Variable
import numpy as np
from ssd import SSD300
from encoder import DataEncoder
from PIL import Image, ImageDraw
import Intersection_Over_Union as iou
import math
import torch.backends.cudnn as cudnn

use_cuda = torch.cuda.is_available() 
# Load model
net = SSD300()
checkpoint = torch.load('checkpoint/Updated_ubear_400_and_Wild_ear_2000_images_model_ssd2.pth')
net.load_state_dict(checkpoint['net'])
net.eval()

if use_cuda:
	print("Gpu is available")
	net = torch.nn.DataParallel(net, device_ids=[0])
	net.cuda()
	cudnn.benchmark = True



iou_path_Ear = open('New_IOU_SSD2_UBEAR.txt', 'w+')

testdata = 'Cropped_UBEAR.txt'
test_All_Data = open(testdata, 'r')
test_File = test_All_Data.readlines()
list_image = []
for linesr in test_File:
    linesr = (linesr.strip()).split(" ")
    line_list = [linesr[0],linesr[2],linesr[3],linesr[4],linesr[5]]
    list_image.append(line_list)
#count = 0
list_image.sort()
for image in list_image:
	#if count == 10:
	    #break   
	#count = count + 1
	try:
	    image_path = '/media/biometric/Data1/Ranjeet/NewPytorch/SSD_Ear_RGB_300/Cropped/' + image[0]
	    img = Image.open(image_path).convert('RGB')
	    img1 = img.resize((300,300))
	    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
	    img1 = transform(img1)
	    if use_cuda:
	        img1 = img1.cuda()
	    loc, conf = net(Variable(img1[None,:,:,:], volatile=True)) # Forward
	    loc = loc.cpu()
	    conf = conf.cpu()
	    print(loc)
	    print(conf)
	    data_encoder = DataEncoder() # Decode
	    boxes, labels, scores = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)
	    #draw = ImageDraw.Draw(img)
	    #draw.rectangle(list(box), outline='blue')
	    #draw.rectangle(ground_truth_box, outline='blue')
	    #img.show()
	    for box in boxes:
	        box[::2] *= img.width
	        box[1::2] *= img.height
	        box = list(box)
	        x1_org = image[1]
	        y1_org = image[2]
	        x2_org = image[3]
	        y2_org = image[4]
	        ground_truth_box =  [int(x1_org),int(y1_org),int(x2_org),int(y2_org)]
	        predict_box = [math.ceil(box[0]),math.ceil(box[1]),math.ceil(box[2]),math.ceil(box[3])]
	        IOU = abs(iou.bb_intersection_over_union(ground_truth_box, predict_box))
	        iou_path_Ear.write(str(IOU) + "\n")     
	except:
	    print("Error")
	    pass
