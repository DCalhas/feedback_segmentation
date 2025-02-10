from feedback_segmentation.data import CirclesDataset, PolygonsDataset

import torch

from torch.utils.data import Dataset

import numpy as np

class ShapesDataset(Dataset):
	"""	
	This class is a combination of the Circles data and the Polygons data

	Example usage:
		>>> shapes=ShapesDataset(10,classes=4)
		>>> shapes.sample()
	"""

	def __init__(self, N, channels=3, classes=5, resolution=(512,512), noise=0.0, irregular_noise=5., dtype=torch.float32):
		assert classes>=4
		assert N%2==0



		self.polygons=PolygonsDataset(int((1/(classes))*N), channels=channels, classes=classes-2, resolution=resolution, noise=noise, irregular_noise=irregular_noise, dtype=dtype)
		self.circles=CirclesDataset(int(((classes-1)/(classes))*N), channels=channels, classes=2, resolution=resolution, noise=noise, dtype=dtype)
		

		self.N=int(((classes-1)/(classes))*N) + int((1/(classes))*N)
		self.resolution=resolution
		self.noise=noise
		self.dtype=dtype
		self.classes=classes
		self.channels=channels
	
		self.data=torch.empty((self.N, )+self.resolution+(self.channels,), dtype=self.dtype,)# 3 for rgb
		self.labels=torch.empty((self.N, )+self.resolution+(self.classes, ), dtype=self.dtype,)

	def sample(self,):

		self.polygons.sample()
		self.circles.sample()

		add_circles=torch.zeros(self.circles.labels.shape[:3]+(self.polygons.labels.shape[-1]-1,), device=self.circles.labels.device, dtype=self.circles.labels.dtype)#background and circles
		add_polygons=torch.zeros(self.polygons.labels.shape[:3]+(self.circles.labels.shape[-1]-1,), device=self.polygons.labels.device, dtype=self.polygons.labels.dtype)#background and circles
		#add classes of circles to polygons and of polygons to circles
		new_polygons_labels=torch.cat([self.polygons.labels[:,:,:,0:1], add_polygons, self.polygons.labels[:,:,:,1:]], dim=3)
		new_circles_labels=torch.cat([self.circles.labels[:,:,:,0:1], self.circles.labels[:,:,:,1:], add_circles,], dim=3)
		
		#concat at the number of instances dimension
		self.data=torch.cat([self.polygons.data, self.circles.data], dim=0)
		self.labels=torch.cat([new_polygons_labels, new_circles_labels], dim=0)
		
	def __getitem__(self, idx):
		x=self.data[idx].permute(2,0,1)+torch.normal(mean=0.0, std=torch.ones((self.channels,)+self.resolution)*self.noise)

		return (x-x.amin())/(x.amax()-x.amin()), self.labels[idx].permute(2,0,1)

	def __len__(self,):

		return self.N
