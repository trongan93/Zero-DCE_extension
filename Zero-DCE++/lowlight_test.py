import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time

def lowlight(image_path):
	os.environ['CUDA_VISIBLE_DEVICES']='0'
	scale_factor = 1
	data_lowlight = Image.open(image_path)

 

	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()

	h=(data_lowlight.shape[0]//scale_factor)*scale_factor
	w=(data_lowlight.shape[1]//scale_factor)*scale_factor
	data_lowlight = data_lowlight[0:h,0:w,:]
	data_lowlight = data_lowlight.permute(2,0,1)
	data_lowlight = data_lowlight.cuda().unsqueeze(0)

	DCE_net = model.enhance_net_nopool(scale_factor).cuda()
	DCE_net.load_state_dict(torch.load('snapshots_Zero_DCE++/Epoch0.pth'))

	# Show the model parameters
	total_params = sum(
		param.numel() for param in DCE_net.parameters()
	)
	print(total_params)
	trainable_params = sum(
		p.numel() for p in DCE_net.parameters() if p.requires_grad
	)
	print(trainable_params)

	# Compuatation complexity of network
	from ptflops import get_model_complexity_info
	macs, params = get_model_complexity_info(DCE_net, (3,h,w), as_strings=True, print_per_layer_stat=True, verbose=True)
	print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
	print('{:<30}  {:<8}'.format('Number of parameters: ', params))

	start = time.time()
	enhanced_image,params_maps = DCE_net(data_lowlight)

	end_time = (time.time() - start)

	print(end_time)
	image_path = image_path.replace('test_data','result_Zero_DCE++')

	result_path = image_path
	if not os.path.exists(image_path.replace('/'+image_path.split("/")[-1],'')):
		os.makedirs(image_path.replace('/'+image_path.split("/")[-1],''))
	# import pdb;pdb.set_trace()
	torchvision.utils.save_image(enhanced_image, result_path)
	return end_time

if __name__ == '__main__':

	# with torch.no_grad():
	#
	# 	filePath = '/mnt/d/Seaport_satellite_images/RGB_Only_Test'
	# 	file_list = os.listdir(filePath)
	# 	sum_time = 0
	# 	for file_name in file_list:
	# 		test_list = glob.glob(filePath+file_name+"/*")
	#
	# 		for image in test_list:
	#
	# 			print(image)
	# 			sum_time = sum_time + lowlight(image)
	#
	# 	print(sum_time)

	with torch.no_grad():

		# filePath = '../ZeroDCE_DataSet/test/'
		filePath = "/mnt/d/SeaportDataset/test_set_4"
		# filePath = "/mnt/d/test_img/1920x1200/8BIT/COLOR"
		# file_list = os.listdir(filePath)
		# print(file_list)

		target_path = './test_output_seaport/set4'
		os.makedirs(target_path, exist_ok=True)
		sum_time = 0
		for file_name in filePath:
			test_list = glob.glob(filePath + file_name + "/*")
			print(test_list)
			os.makedirs(target_path + file_name, exist_ok=True)
			for image in test_list:
				print(image)
				sum_time = sum_time + lowlight(image)

		print(sum_time)


