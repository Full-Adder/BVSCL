import torch
from torch.distributed.nn import all_gather

from .saliency_db import saliency_db
from .transforms import TemporalRandomCrop, SpatialTransform, SpatialTransform_norm

import os
from os.path import join
import csv
import cv2, copy
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
# import torchaudio
import sys
from scipy.io import wavfile
import json


class DHF1KDataset(Dataset):
	def __init__(self, path_data, len_snippet, mode="train", multi_frame=0, alternate=1):
		self.path_data = path_data
		self.len_snippet = len_snippet
		self.mode = mode
		self.multi_frame = multi_frame
		self.alternate = alternate
		self.img_transform = transforms.Compose([
			transforms.Resize((224, 384)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])
		if self.mode == "train":
			self.video_names = os.listdir(path_data)
			self.list_num_frame = [len(os.listdir(os.path.join(path_data, d, 'images'))) for d in self.video_names]
		elif self.mode == "val":
			self.list_num_frame = []
			for v in os.listdir(path_data):
				for i in range(0, len(
						os.listdir(os.path.join(path_data, v, 'images'))) - self.alternate * self.len_snippet,
							   32):
					self.list_num_frame.append((v, i))
		else:
			self.list_num_frame = []
			for v in os.listdir(path_data):
				for i in range(0, len(
						os.listdir(os.path.join(path_data, v, 'images'))) - self.alternate * self.len_snippet,
							   self.len_snippet):
					self.list_num_frame.append((v, i))
				self.list_num_frame.append(
					(v, len(os.listdir(os.path.join(path_data, v, 'images'))) - self.len_snippet))

	def __len__(self):
		return len(self.list_num_frame)

	def __getitem__(self, idx):
		if self.mode == "train":
			file_name = self.video_names[idx]
			start_idx = np.random.randint(0, self.list_num_frame[idx] - self.alternate * self.len_snippet + 1)
		elif self.mode == "val" or self.mode == "save":
			(file_name, start_idx) = self.list_num_frame[idx]

		path_clip = os.path.join(self.path_data, file_name, 'images')
		path_annt = os.path.join(self.path_data, file_name, 'maps')

		clip_img = []
		clip_gt = []

		for i in range(self.len_snippet):
			img = Image.open(os.path.join(path_clip, '%04d.png' % (start_idx + self.alternate * i + 1))).convert('RGB')
			sz = img.size

			if self.mode != "save":
				gt = np.array(
					Image.open(os.path.join(path_annt, '%04d.png' % (start_idx + self.alternate * i + 1))).convert('L'))
				gt = gt.astype('float')

				if self.mode == "train":
					gt = cv2.resize(gt, (384, 224))

				if np.max(gt) > 1.0:
					gt = gt / 255.0
				clip_gt.append(torch.FloatTensor(gt))

			clip_img.append(self.img_transform(img))

		clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
		if self.mode != "save":
			clip_gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))
		if self.mode == "save":
			return clip_img, start_idx, file_name, sz
		else:
			if self.multi_frame == 0:
				return clip_img, clip_gt[-1]
			return clip_img, clip_gt


class Hollywood_UCFDataset(Dataset):
	def __init__(self, path_data, len_snippet, mode="train", frame_no="last", multi_frame=0):
		self.path_data = path_data
		self.len_snippet = len_snippet
		self.mode = mode
		self.frame_no = frame_no
		self.multi_frame = multi_frame
		self.img_transform = transforms.Compose([
			transforms.Resize((224, 384)),
			transforms.ToTensor(),
			transforms.Normalize(
				[0.485, 0.456, 0.406],
				[0.229, 0.224, 0.225]
			)
		])
		if self.mode == "train":
			self.video_names = os.listdir(path_data)
			self.list_num_frame = [len(os.listdir(os.path.join(path_data, d, 'images'))) for d in self.video_names]
		elif self.mode == "val":
			self.list_num_frame = []
			for v in os.listdir(path_data):
				for i in range(0, len(os.listdir(os.path.join(path_data, v, 'images'))) - self.len_snippet,
							   self.len_snippet):
					self.list_num_frame.append((v, i))
				if len(os.listdir(os.path.join(path_data, v, 'images'))) <= self.len_snippet:
					self.list_num_frame.append((v, 0))

	def __len__(self):
		return len(self.list_num_frame)

	def __getitem__(self, idx):
		if self.mode == "train":
			file_name = self.video_names[idx]
			start_idx = np.random.randint(0, max(1, self.list_num_frame[idx] - self.len_snippet + 1))
		elif self.mode == "val":
			(file_name, start_idx) = self.list_num_frame[idx]

		path_clip = os.path.join(self.path_data, file_name, 'images')
		path_annt = os.path.join(self.path_data, file_name, 'maps')

		clip_img = []
		clip_gt = []

		list_clips = os.listdir(path_clip)
		list_clips.sort()
		list_sal_clips = os.listdir(path_annt)
		list_sal_clips.sort()

		if len(list_sal_clips) < self.len_snippet:
			temp = [list_clips[0] for _ in range(self.len_snippet - len(list_clips))]
			temp.extend(list_clips)
			list_clips = copy.deepcopy(temp)

			temp = [list_sal_clips[0] for _ in range(self.len_snippet - len(list_sal_clips))]
			temp.extend(list_sal_clips)
			list_sal_clips = copy.deepcopy(temp)

			assert len(list_sal_clips) == self.len_snippet and len(list_clips) == self.len_snippet

		for i in range(self.len_snippet):
			img = Image.open(os.path.join(path_clip, list_clips[start_idx + i])).convert('RGB')
			clip_img.append(self.img_transform(img))

			gt = np.array(Image.open(os.path.join(path_annt, list_sal_clips[start_idx + i])).convert('L'))
			gt = gt.astype('float')

			if self.mode == "train":
				gt = cv2.resize(gt, (384, 224))

			if np.max(gt) > 1.0:
				gt = gt / 255.0
			clip_gt.append(torch.FloatTensor(gt))

		clip_img = torch.FloatTensor(torch.stack(clip_img, dim=0))
		if self.multi_frame == 0:
			gt = clip_gt[-1]
		else:
			gt = torch.FloatTensor(torch.stack(clip_gt, dim=0))

		return clip_img, gt

def get_dataloader(root:str, mode:str, task:str,
				   datasetName_list:list=None, 
				   with_audio:bool=True,
				   batch_size:int=1, num_workers:int=0,
				   sample_size:int=(384,224),			# 输入尺寸
				   sample_duration:int=32,		# 最终采样长度
				   step_duration:int=90,		# 空余采样长度
				   spatial_transform=None,
				   spatial_transform_norm=None,
				   temporal_transform=None,
				   pin_memory:bool=True,
				   infoBatch_epoch:int=-1,		# is use infoBatch
				   ):
	
	assert mode in ['train', 'val', 'test'], f"mode should be 'train', 'val' or 'test', but got {mode}"
	assert task in ['class_increase', 'domain_increase', 'joint'], f"task should be 'class_increase' or 'domain_increase' or 'joint', but got {task}"
	
	if task == 'domain_increase':
		if datasetName_list == None:
			# datasetName_list = ['Coutrot_db1', 'Coutrot_db2', 'SumMe', 'ETMD_av', 'AVAD', 'DHKF-1k', 'Hollywood']	# miss DIEM
			datasetName_list = ['AVAD',"AVAD"]	# miss DIEM
			# 由于每个任务需要的学习轮数不同，需要手动配置训练的数据集名称和顺序，选择训练较好的权重，进行持续训练。
		else:
			for i in datasetName_list:
				assert i in ['Coutrot_db1', 'Coutrot_db2', 'SumMe', 'ETMD_av', 'AVAD'], f"datasetName_list should be ['DIEM', 'Coutrot_db1', 'Coutrot_db2', 'SumMe', 'ETMD_av', 'AVAD'], but got {i}"
	elif task == 'class_increase':
		if datasetName_list == None:
			datasetName_list = ['doing', 'nature', 'society', 'something', 'talk']
		else:
			for i in datasetName_list:
				assert i in ['doing', 'nature', 'society', 'something', 'talk'], f"datasetName_list should be ['doing', 'nature', 'society', 'something', 'talk'], but got {i}"
	elif task == 'joint':
		if datasetName_list == None:
			datasetName_list = ['DIEM', 'Coutrot_db1', 'Coutrot_db2', 'SumMe', 'ETMD_av', 'AVAD', 'DHKF-1k', 'Hollywood']
		else:
			for i in datasetName_list:
				assert i in ['DIEM', 'Coutrot_db1', 'Coutrot_db2', 'SumMe', 'ETMD_av', 'AVAD','DHKF-1k', 'Hollywood'], f"datasetName_list should be ['DIEM', 'Coutrot_db1', 'Coutrot_db2', 'SumMe', 'ETMD_av', 'AVAD'], but got {i}"

	if spatial_transform == None:
		spatial_transform = SpatialTransform(mode, sample_size)
	if spatial_transform_norm == None:
		spatial_transform_norm = SpatialTransform_norm((0.3818, 0.3678, 0.3220), (0.2727, 0.2602, 0.2568))
	if temporal_transform == None:
		temporal_transform = TemporalRandomCrop(sample_duration)


	txt_mode = 'train' if mode == 'train' else 'test' 
	all_dataset = []
  
	for dataset_name in datasetName_list:
		print(f"load dataset: {dataset_name}({mode})...")

		if task == 'domain_increase' or task == 'joint':
			training_data = saliency_db(
				f'{root}/video_frames/{dataset_name}/',
				f'{root}/fold_lists/{dataset_name}_list_{txt_mode}_1_fps.txt',
				f'{root}/annotations/{dataset_name}/',
				f'{root}/video_audio/{dataset_name}/',
				with_audio=with_audio,
				spatial_transform=spatial_transform,
				spatial_transform_norm=spatial_transform_norm,
				temporal_transform= temporal_transform,
				exhaustive_sampling=(mode == 'test'),
				sample_duration=sample_duration,	# 采样长度
				step_duration=step_duration,		# 窗口长度
				)
			all_dataset.append(training_data)

		elif task == 'class_increase':
			training_data = saliency_db(
				f'{root}/video_frames/',
				f'{root}/class_category/{dataset_name}_{txt_mode}.txt',
				f'{root}/annotations/',
				f'{root}/video_audio/',
				spatial_transform=spatial_transform,
				spatial_transform_norm=spatial_transform_norm,
				temporal_transform= temporal_transform,
				exhaustive_sampling=(mode == 'test'),
				sample_duration=sample_duration,	# 采样长度
				step_duration=step_duration,		# 窗口长度
				)
			all_dataset.append(training_data)

		if task == 'joint':
			data_set = torch.utils.data.ConcatDataset(all_dataset)
			all_dataset = [data_set]

	dataloder_list = []
	for data_set in all_dataset:
		if infoBatch_epoch>0:		# TODO: need to test
			from .utils.InfoBatch import InfoBatch
			data_set = InfoBatch(data_set, num_epochs=infoBatch_epoch)
			print("*** use infoBatch! Remember use \"loss_fun = nn.CrossEntropyLoss(reduction='none')\" and \"loss = dataset.update(loss)\" in training loop")

		data_loader = torch.utils.data.DataLoader(
			data_set,
			batch_size=batch_size,
			num_workers=num_workers,
			shuffle=(mode == 'train' and infoBatch_epoch<=0),
			pin_memory=pin_memory,
			drop_last=(mode == 'train'),
			sampler = data_set.sampler if infoBatch_epoch>0 else None)
		dataloder_list.append(data_loader)

	if infoBatch_epoch>0:
		return dataloder_list, all_dataset, datasetName_list
	else:
		return dataloder_list, datasetName_list if task != 'joint' else ["joint"]
