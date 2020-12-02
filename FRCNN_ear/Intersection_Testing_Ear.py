import os
import cv2
import numpy as np
import sys
import pickle
from optparse import OptionParser
import time
from keras_frcnn import config
import keras_frcnn.resnet as nn
from keras import backend as K
from keras.layers import Input
from keras.models import Model
from keras_frcnn import roi_helpers
import Intersection_Over_Union as iou
sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("-p", "--path", dest="test_path", help="Path to test data.")
parser.add_option("-n", "--num_rois", dest="num_rois",
				help="Number of ROIs per iteration. Higher means more memory use.", default=8)
parser.add_option("--config_filename", dest="config_filename", help=
				"Location to read the metadata related to the training (generated when training).",
				default="config.pickle")

(options, args) = parser.parse_args()

if not options.test_path:   # if filename is not given
	parser.error('Error: path to test data must be specified. Pass --path to command line')


config_output_filename = options.config_filename

with open(config_output_filename, 'r') as f_in:
	C = pickle.load(f_in)

# turn off any data augmentation at test time
C.use_horizontal_flips = False
C.use_vertical_flips = False
C.rot_90 = False

img_path = options.test_path

Accuracy = []

def format_img(img, C):
	img_min_side = float(C.im_size)
	(height,width,_) = img.shape
	
	if width <= height:
		f = img_min_side/width
		new_height = int(f * height)
		new_width = int(img_min_side)
	else:
		f = img_min_side/height
		new_width = int(f * width)
		new_height = int(img_min_side)
	img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
	img = img[:, :, (2, 1, 0)]
	img = img.astype(np.float32)
	img[:, :, 0] -= C.img_channel_mean[0]
	img[:, :, 1] -= C.img_channel_mean[1]
	img[:, :, 2] -= C.img_channel_mean[2]
	img /= C.img_scaling_factor
	img = np.transpose(img, (2, 0, 1))
	img = np.expand_dims(img, axis=0)
	return img


class_mapping = C.class_mapping

if 'bg' not in class_mapping:
	class_mapping['bg'] = len(class_mapping)

class_mapping = {v: k for k, v in class_mapping.iteritems()}
#print(class_mapping)
class_to_color = {class_mapping[v]: np.random.randint(0, 255, 3) for v in class_mapping}
C.num_rois = int(options.num_rois)

if K.image_dim_ordering() == 'th':
	input_shape_img = (3, None, None)
	input_shape_features = (1024, None, None)
else:
	input_shape_img = (None, None, 3)
	input_shape_features = (None, None, 1024)

totaliou=0
img_input = Input(shape=input_shape_img)
roi_input = Input(shape=(C.num_rois, 4))
feature_map_input = Input(shape=input_shape_features)

# define the base network (resnet here, can be VGG, Inception, etc)
shared_layers = nn.nn_base(img_input, trainable=True)

# define the RPN, built on the base layers
num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios)
rpn_layers = nn.rpn(shared_layers, num_anchors)

classifier = nn.classifier(feature_map_input, roi_input, C.num_rois, nb_classes=len(class_mapping), trainable=True)

model_rpn = Model(img_input, rpn_layers)
model_classifier_only = Model([feature_map_input, roi_input], classifier)

model_classifier = Model([feature_map_input, roi_input], classifier)

model_rpn.load_weights(C.model_path, by_name=True)
model_classifier.load_weights(C.model_path, by_name=True)

model_rpn.compile(optimizer='sgd', loss='mse')
model_classifier.compile(optimizer='sgd', loss='mse')

all_imgs = []

classes = {}

bbox_threshold = .70
count=0
i=0
iou_path = open('IOU_on.txt', 'wb+')
visualise = True
#path_Rect_txt = ''
fo = open('training1.txt' , "r")
#pl = fo.readlines(fo)
for line in fo:
	line_split = line.strip().split(',')
	(filename,x_org,y_org,w_org,h_org,class_name) = line_split
	for idx, img_name in enumerate(sorted(os.listdir(img_path))):
		if not img_name.lower().endswith(('.bmp', '.jpeg', '.jpg', '.png', '.tif', '.tiff')):
			continue
	#print(img_name)
	#print (idx)
		st = time.time()
		filepath = os.path.join(img_path,filename)
		filename1 = filepath
		if filepath==filename1:
			img = cv2.imread(filepath)
			height_org = np.size(img, 0)
			width_org = np.size(img, 1)
			print height_org
			X = format_img(img, C)
 # Code for getting ground truth coordinate
			#path_Rect_txt = '/home/biometric/Ranjeet/Data_For_Faster_RCNN/knuckle_roi/coordinates'
			p = img_name.split('.')
			p1 = p[0]
			#file1 = p1 + '.txt'
			#print p
			#fo = open(path_Rect_txt + '/' + file1 , "r")
			#line = fo.readlines()
			#x_org,y_org,w_org,h_org = (line[1].strip()).split(" ")
	
			print i
			i=i+1
			#img_scaled = np.transpose(X.copy()[0, (2, 1, 0), :, :], (1, 2, 0)).copy()
			img_scaled[:, :, 0] += 123.68
			img_scaled[:, :, 1] += 116.779
			img_scaled[:, :, 2] += 103.939
	
			img_scaled = img_scaled.astype(np.uint8)
			height_new = np.size(img_scaled, 0)
			width_new = np.size(img_scaled, 1)
	
			# ratio between new and original image 
			ratio_h = float(height_new)/height_org
			ratio_w = float(width_new)/width_org
			#print ratio_h
			#print ratio_w
        
			if K.image_dim_ordering() == 'tf':
				X = np.transpose(X, (0, 2, 3, 1))

			# get the feature maps and output from the RPN
			[Y1, Y2, F] = model_rpn.predict(X)

			R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_dim_ordering(),overlap_thresh=0.7)

			# convert from (x1,y1,x2,y2) to (x,y,w,h)
			R[:, 2] -= R[:, 0]
			R[:, 3] -= R[:, 1]

			# apply the spatial pyramid pooling to the proposed regions
			bboxes = {}
			probs = {}

			for jk in range(R.shape[0]//C.num_rois + 1):
				ROIs = np.expand_dims(R[C.num_rois*jk:C.num_rois*(jk+1), :], axis=0)
				if ROIs.shape[1] == 0:
					break

				if jk == R.shape[0]//C.num_rois:
					#pad R
					curr_shape = ROIs.shape
					target_shape = (curr_shape[0],C.num_rois,curr_shape[2])
					ROIs_padded = np.zeros(target_shape).astype(ROIs.dtype)
					ROIs_padded[:, :curr_shape[1], :] = ROIs
					ROIs_padded[0, curr_shape[1]:, :] = ROIs[0, 0, :]
					ROIs = ROIs_padded

				[P_cls, P_regr] = model_classifier_only.predict([F, ROIs])

				for ii in range(P_cls.shape[1]):

					if np.max(P_cls[0, ii, :]) < bbox_threshold or np.argmax(P_cls[0, ii, :]) == (P_cls.shape[2] - 1):
						continue

					cls_name = class_mapping[np.argmax(P_cls[0, ii, :])]

					if cls_name not in bboxes:
						bboxes[cls_name] = []
						probs[cls_name] = []

					(x, y, w, h) = ROIs[0, ii, :]

					cls_num = np.argmax(P_cls[0, ii, :])
					try:
						(tx, ty, tw, th) = P_regr[0, ii, 4*cls_num:4*(cls_num+1)]
						tx /= C.classifier_regr_std[0]
						ty /= C.classifier_regr_std[1]
						tw /= C.classifier_regr_std[2]
						th /= C.classifier_regr_std[3]
						x, y, w, h = roi_helpers.apply_regr(x, y, w, h, tx, ty, tw, th)
					except:
						pass
					bboxes[cls_name].append([16*x, 16*y, 16*(x+w), 16*(y+h)])
					probs[cls_name].append(np.max(P_cls[0, ii, :]))
		
	
			# Finding coordinate of box with highest probability among all the boxes
			prob = 0
			prob_cordinate = [0,0,0,0]
			all_dets = []
			#print bboxes
			for key in bboxes :
				bbox = np.array(bboxes[key])
				new_boxes, new_probs = roi_helpers.non_max_suppression_fast(bbox, np.array(probs[key]), overlap_thresh=0.5)
				new_probs = list(new_probs )
				max_prob = max(new_probs)
				element_index = new_probs.index(max_prob)
				if(max_prob>prob):
					prob = max_prob
					prob_cordinate = new_boxes[element_index,:]
				count=count+1
		
		
			(x1, y1, x2, y2) = prob_cordinate
	
			x1_org  = float(x_org) * ratio_h
			y1_org  = float(y_org) * ratio_w
			w_org  = float(float(w_org)) * ratio_h
			h_org = float(float(h_org)) * ratio_w
			x2_org = w_org #+ x1_org
			y2_org = h_org #+ y1_org
			x1_org  = int(x1_org)
			y1_org = int(y1_org)
			x2_org = int(x2_org)
			y2_org  =int(y2_org)
				
			ground_truth_box =  [x1_org,y1_org,x2_org,y2_org]
			predict_box = [x1,y1,x2,y2]
			# Find Intersection over Union
			IOU = abs(iou.bb_intersection_over_union(ground_truth_box, predict_box))
			#print "IOU = " + str(IOU)
				
			iou_path.write(str(IOU) + "\n")
			print IOU
			totaliou=totaliou+1
			#cv2.rectangle(img_scaled,(x1, y1), (x2, y2), class_to_color[key],2)
			#cv2.rectangle(img_scaled,(x1_org, y1_org), (x2_org, y2_org), (123,120,255))
			#textLabel = '{}: {}'.format(key,int(100*prob))
			#all_dets.append((key,100*prob))
			#print new_probs
			#(retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,1,1)
			#textOrg = (x1, y1+20)
			
		#cv2.rectangle(img_scaled, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
		#cv2.rectangle(img_scaled, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
		#cv2.putText(img_scaled, textLabel, textOrg, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
				
		#cv2.imshow('img', img_scaled)
		#cv2.waitKey(0)
		#print(all_dets)
	
	#Final_Accuracy = float(sum(Accuracy))/len(Accuracy)
	#print " Accuracy = " + str(Final_Accuracy)
	
	
	#cv2.imshow('img', img_scaled)
	#cv2.waitKey(0)	
	
print count
print totaliou	

