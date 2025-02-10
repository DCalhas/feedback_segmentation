from torch.utils.data import Dataset

import torch 

import numpy as np

import cv2

class CirclesDataset(Dataset):
	"""
	This class generates a N number of circles with various radius from 1/50 of the resolution to 1/4

	Example:
	>>> circles=CirclesDataset(100,)
	>>> circles.sample()
	"""

	def __init__(self, N, channels=3, classes=2, resolution=(512,512), noise=0.0, dtype=torch.float32):

		self.N=N
		self.resolution=resolution
		self.noise=noise
		self.dtype=dtype
		self.classes=classes
		self.channels=channels
	
		self.data=torch.empty((self.N, )+self.resolution+(self.channels,), dtype=self.dtype,)# 3 for rgb
		self.labels=torch.empty((self.N, )+self.resolution+(self.classes, ), dtype=self.dtype,)

	def sample(self,):


		identity=np.eye(2,)

		for n in range(self.N):


			xi=np.random.randint(0, high=self.resolution[0])
			yi=np.random.randint(0, high=self.resolution[1])

			radius=np.random.uniform(low=self.resolution[0]/50, high=self.resolution[0]/4)

			H, W=np.ogrid[:self.resolution[0], :self.resolution[1]]
			dist_from_center = np.sqrt((H - xi)**2 + (W-yi)**2)
			im = (dist_from_center <= radius).astype(np.float32)
			
			label=np.zeros(self.resolution+(2,), dtype=np.float32)
			label=(im>0.).astype(np.int32)

			label=identity[label]
			im=cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
			
			self.data[n]=torch.from_numpy(im).to(dtype=self.dtype)
			self.labels[n]=torch.from_numpy(label).to(dtype=self.dtype)

	def __getitem__(self, idx):
		x=self.data[idx].permute(2,0,1)+torch.normal(mean=0.0, std=torch.ones((self.channels,)+self.resolution)*self.noise)

		return (x-x.amin())/(x.amax()-x.amin()), self.labels[idx].permute(2,0,1)

	def __len__(self,):

		return self.N
