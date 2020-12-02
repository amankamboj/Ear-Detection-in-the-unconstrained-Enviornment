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
# Load model

use_cuda = torch.cuda.is_available() 


net = SSD300()
checkpoint = torch.load('/media/biometric/Data1/Ranjeet/NewPytorch/SSD_Ear_RGB_300/Model/Model_Org.pth')
net.load_state_dict(checkpoint['net'])
net.eval()
ground_path_Ear = open('Mis_Cropped_UBEAR1.txt', 'w+')

if use_cuda:
	print("Gpu is available")
	net = torch.nn.DataParallel(net, device_ids=[0])
	net.cuda()
	cudnn.benchmark = True

testdata = 'new_dataUBEAR2.txt'
test_All_Data = open(testdata, 'r')
test_File = test_All_Data.readlines()
list_image = []
for linesr in test_File:
	linesr = (linesr.strip()).split(" ")
	line_list = [linesr[0],linesr[2],linesr[3],linesr[4],linesr[5]]
	list_image.append(line_list)
#print(list_image)
#count = 0

for image in list_image:
		# if count == 10:
		# 	break   
		# count = count + 1
	try:
		image_path = '/media/biometric/Data1/Ranjeet/NewPytorch/Challenge_images/' + image[0]
		if os.path.exists(image_path):
			img = Image.open(image_path).convert("RGB")
			img_resize = img.resize((300,300))
			transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.485,), std=(0.229,))])
			img_transform = transform(img_resize)
			if use_cuda:
				img1 = img_transform.cuda()
			loc, conf = net(Variable(img1[None,:,:,:], volatile=True)) # Forward
			loc = loc.cpu()
			conf = conf.cpu()
			data_encoder = DataEncoder() # Decode
		#print(loc)
			boxes, labels, scores = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)
			for box in boxes:
				box[::2] *= img.width
				box[1::2] *= img.height
				box = list(box)
				x1_org_old = float(image[1])
				y1_org_old = float(image[2])
				x2_org_old = float(image[3])
				y2_org_old = float(image[4])
				ground_truth_box =  [int(x1_org_old),int(y1_org_old),int(x2_org_old),int(y2_org_old)]
				predict_box = [math.ceil(box[0]),math.ceil(box[1]),math.ceil(box[2]),math.ceil(box[3])]
				IOU = abs(iou.bb_intersection_over_union(ground_truth_box, predict_box))
				if IOU>0.8:
					box[0] =  box[0] - 50 
					box[1] =  box[1] - 50
					box[2] =  box[2] + 50
					box[3] =  box[3] + 50
					x1_org = float(image[1]) - box[0]
					y1_org = float(image[2]) - box[1]
					x2_org = float(image[3]) - box[0]
					y2_org = float(image[4]) - box[1]
					cropped_img = img.crop(box)
					draw = ImageDraw.Draw(cropped_img)
					path_Ear = './New_cropped1/' + image[0]
					cropped_img.save(path_Ear)
					ground_path_Ear.write(image[0]+ " " + "1" + " " + str(int(x1_org)) + " " + str(int(y1_org)) + " " + str(int(x2_org)) + " " + str(int(y2_org)) + " " + str(0) + "\n") 
				else:
					x1_org = float(image[1]) - 50 
					y1_org = float(image[2]) - 50 
					x2_org = float(image[3]) + 50 
					y2_org = float(image[4]) + 50 
					new_box = [x1_org, y1_org, x2_org, y2_org]  
					cropped_img = img.crop(new_box )
					draw = ImageDraw.Draw(cropped_img)
					x1_new = float(image[1]) - x1_org
					y1_new = float(image[2]) - y1_org 
					x2_new = float(image[3]) - x1_org
					y2_new = float(image[4]) - y1_org 
					path_Ear = './New_cropped1/' + image[0]
					cropped_img.save(path_Ear)
					ground_path_Ear.write(image[0]+ " " + "1" + " " + str(int(x1_new)) + " " + str(int(y1_new)) + " " + str(int(x2_new)) + " " + str(int(y2_new)) + " " + str(0) + "\n") 
			print("exists")
		else:
			print("no path exist")    
	except:
		x1_org = float(image[1]) - 50 
		y1_org = float(image[2]) - 50 
		x2_org = float(image[3]) + 50 
		y2_org = float(image[4]) + 50 
		new_box = [x1_org, y1_org, x2_org, y2_org]  
		cropped_img = img.crop(new_box)
		draw = ImageDraw.Draw(cropped_img)
		x1_new = float(image[1]) - x1_org
		y1_new = float(image[2]) - y1_org 
		x2_new = float(image[3]) - x1_org
		y2_new = float(image[4]) - y1_org 
		path_Ear = './New_cropped1/' + image[0]
		cropped_img.save(path_Ear)
		ground_path_Ear.write(image[0]+ " " + "1" + " " + str(int(x1_new)) + " " + str(int(y1_new)) + " " + str(int(x2_new)) + " " + str(int(y2_new)) + " " + str(0) + "\n") 
