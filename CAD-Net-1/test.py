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
import torch.backends.cudnn as cudnn
use_cuda = torch.cuda.is_available() 

# Load model
net = SSD300()
#net = net.cuda() # for gpu
checkpoint = torch.load('checkpoint/ubear_400_images_model.pth')
net.load_state_dict(checkpoint['net'])
net.eval()



if use_cuda:
	print("Gpu is available")
	net = torch.nn.DataParallel(net, device_ids=[0])
	net.cuda()
	cudnn.benchmark = True

# Load test image
list_image = os.listdir('/media/biometric/Data1/Database/Ear_DataSet/UBEAR/Data2/All_tiff')
for image in list_image:
	
	image_path = '/media/biometric/Data1/Database/Ear_DataSet/UBEAR/Data2/All_tiff/' + image

	#img = Image.open(image_path)
	print("Ranjeet")
	#print(img)		
	img = Image.open(image_path).convert('RGB')
	print(img)
	img1 = img.resize((300,300))
	transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
	img1 = transform(img1)
	if use_cuda:
	    img1 = img1.cuda()
	# Forward
	loc, conf = net(Variable(img1[None,:,:,:], volatile=True))
	loc = loc.cpu()
	conf = conf.cpu()
	# Decode


	data_encoder = DataEncoder()
	print("Ranjeet Hello")
	boxes, labels, scores = data_encoder.decode(loc.data.squeeze(0), F.softmax(conf.squeeze(0)).data)
	image_path = 'Testing_New/' + image
	draw = ImageDraw.Draw(img)
	for box in boxes:
	    box[::2] *= img.width
	    box[1::2] *= img.height
	    draw.rectangle(list(box), outline='blue',width=5)
	img.save(image_path)

	print("error")


    	
