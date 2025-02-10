from torch.utils.data import Dataset

import torch

from skimage.draw import polygon

import cv2

import numpy as np

import random

import torchvision

from PIL import Image, ImageDraw


def generate_polygon(N, radius):
	"""
	Given a N, number of sides that is > 3 and the radius
	this function returns the vertices of a polygon
	"""
	vertices=np.empty((2,N), np.int32)	
	u=np.ones((2,), dtype=np.float32)
	u[0]=0.

	angle=0
	for n in range(N):
		_angle=angle+2*np.pi*(n/N)
		R=np.array([[np.cos(_angle), -np.sin(_angle)], [np.sin(_angle),  np.cos(_angle)]], dtype=np.float32)
		vertices[:,n] = (radius[n]*np.dot(u, R)).astype(np.int32)

	return vertices

class PolygonsDataset(Dataset):
	"""
	This class generates a N number of polygons with various random side length

	Example:
	>>> polygons=PolygonsDataset(100,)
	>>> polygons.sample()
	"""

	def __init__(self, N, channels=3, classes=5, resolution=(512,512), noise=0.0, irregular_noise=5., dtype=torch.float32):
		assert classes>0
		
		self.N=N
		self.resolution=resolution
		self.noise=noise
		self.irregular_noise=irregular_noise
		self.dtype=dtype
		self.classes=classes
		self.channels=channels
	
		self.data=torch.empty((self.N, )+self.resolution+(self.channels,), dtype=self.dtype,)# 3 for rgb
		self.labels=torch.empty((self.N, )+self.resolution+(self.classes+1, ), dtype=self.dtype,)

	def sample(self,):


		identity=np.eye(self.classes+1,)

		for n in range(self.N):
			_class=np.random.randint(3, high=self.classes+3)
			radius=np.random.uniform(low=self.resolution[0]/10, high=self.resolution[0]/4)
			vertices=generate_polygon(_class,np.abs(np.random.normal(loc=radius, scale=self.irregular_noise, size=(_class,))))
			
			xi=np.random.randint(0, high=self.resolution[0])
			yi=np.random.randint(0, high=self.resolution[1])
			vertices[0]+=xi
			vertices[1]+=yi
			im=np.zeros(self.resolution+(1,), dtype=np.float32)
			im[polygon(vertices[0], vertices[1], shape=im.shape)]=1.
			label=np.zeros(self.resolution+(2,), dtype=np.float32)
			label=(im>0.).astype(np.int32)*(_class-2)

			label=identity[label].squeeze(2)#shift minus 2 to ensure triangle is first class and zero is background
			im=cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
			
			self.data[n]=torch.from_numpy(im).to(dtype=self.dtype)
			self.labels[n]=torch.from_numpy(label).to(dtype=self.dtype)

	def __getitem__(self, idx):
		x=self.data[idx].permute(2,0,1)+torch.normal(mean=0.0, std=torch.ones((self.channels,)+self.resolution)*self.noise)

		return (x-x.amin())/(x.amax()-x.amin()), self.labels[idx].permute(2,0,1)

	def __len__(self,):

		return self.N


class OneInMultiplePolygons(Dataset):
	"""	
	This class is a combination of multiple polygons, with only one class being correct - the circles
	""" 
	#Inits the variables
	def __init__(self, n_samples = 1, channels = 3, classes = 4, resolution = (512,512), noise = 0.0, dtype=torch.float32):

		self.N = n_samples
		self.resolution=resolution
		self.noise=noise
		self.dtype=dtype
		self.classes=classes
		self.channels=channels

		self.transform = torchvision.transforms.Compose([
				torchvision.transforms.ToTensor()
			])

		self.data=torch.empty((self.N, )+self.resolution+(self.channels,), dtype=self.dtype,)# 3 for rgb
		self.labels=torch.empty((self.N, )+self.resolution+(self.classes, ), dtype=self.dtype,)

	#Builts the dataset
	def sample(self):
		

		if self.classes == 4:
			print("Sampling the dataset...")
		else:
			Exception("This script only works with 4 classes!")
			
		for sample in range(self.N):
			self.colors = ["red", "blue","pink", "green", "yellow","purple"]
			# Test to asses how this all works
			# Create a blank image

			width, height = self.resolution
			#Image init
			image = Image.new("RGB", (width, height), "black")
			draw = ImageDraw.Draw(image)

			#masks init
			bg_mask = Image.new("L", (width, height), "white")
			sq_mask = Image.new("L", (width, height), "black")
			tr_mask = Image.new("L", (width, height), "black")
			cl_mask = Image.new("L", (width, height), "black")

			bg_mask_draw = ImageDraw.Draw(bg_mask)
			sq_mask_draw = ImageDraw.Draw(sq_mask)
			tr_mask_draw = ImageDraw.Draw(tr_mask)
			cl_mask_draw = ImageDraw.Draw(cl_mask)

			

			num_squares = random.randint(0,2)

			#First generate the squares
			for i in range(num_squares):
				x = random.randint(5, self.resolution[0]-50)
				y = random.randint(5, self.resolution[1]-50)
				radius = random.randint(1,self.resolution[0]/2)
				rot = random.randint(0, 360)
				my_color = random.randint(0, len(self.colors)-1)
				
				draw.regular_polygon(bounding_circle = (x,y,radius),n_sides = 4 , rotation = rot, fill = self.colors[my_color], )
				
				sq_mask_draw.regular_polygon(bounding_circle = (x,y,radius),n_sides = 4 , rotation = rot, fill = "white", )
				
				del self.colors[my_color]


			#Than, generate the triangles over
			num_triangles = random.randint(1,2)

			for i in range(num_triangles):
				x = random.randint(5, self.resolution[0]-50)
				y = random.randint(5, self.resolution[1]-50)
				radius=np.random.uniform(low=self.resolution[0]/10, high=self.resolution[0]/4)
				rot = random.randint(0, 360)
				my_color = random.randint(0, len(self.colors)-1)

				draw.regular_polygon(bounding_circle = (x,y,radius),n_sides = 3 , rotation = rot, fill = self.colors[my_color], )
				
				tr_mask_draw.regular_polygon(bounding_circle = (x,y,radius),n_sides = 3 , rotation = rot, fill = "white", )

				del self.colors[my_color]

			#Finally we will generate the circles 
			num_circles = random.randint(1,2)
			
			for i in range(num_circles):
				
				x1 = random.randint(2, self.resolution[0]-50)
				y1 = random.randint(2, self.resolution[1]-50)
				x2 = random.randint(10, self.resolution[0]/2)
				y2 = random.randint(10, self.resolution[1]/2)
				radius = random.randint(1, self.resolution[0]/2)
				my_color = random.randint(0, len(self.colors)-1)

				draw.ellipse(xy= [(x1,y1), (x1 + x2,y1+y2)], fill = self.colors[my_color])
				
				cl_mask_draw.ellipse(xy= [(x1,y1), (x1 + x2,y1+y2)])

				del self.colors[my_color]
			
			#Transform into tensor and
			tensor_image = self.transform(image).permute(1,2,0)

			#Mask to torch 
			tensor_sq, tensor_tr, tensor_cl, tensor_bg = self.transform(sq_mask).squeeze(),self.transform(tr_mask).squeeze(), self.transform(cl_mask).squeeze(), self.transform(bg_mask).squeeze()
			
			#Adapt the masks based on the location of each object
			tensor_tr = torch.where(tensor_cl == 1, 0, tensor_tr)
			tensor_sq = torch.where((tensor_cl ==1) | (tensor_tr == 1), 0, tensor_sq)
			tensor_bg = torch.where((tensor_sq == 0) & (tensor_tr == 0) & (tensor_cl == 0), 1, 0)
			
			all_masks = torch.stack([tensor_bg, tensor_tr], dim=2)

			self.data[sample] = tensor_image
			self.labels[sample] = all_masks

	def __len__(self,):
	
		return self.N
	
	def __getitem__(self, idx):

		return self.data[idx].permute(2,0,1)+torch.normal(mean=0.0, std=torch.ones((self.channels,)+self.resolution)*self.noise), self.labels[idx].permute(2,0,1)
