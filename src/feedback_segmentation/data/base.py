from torch.utils.data import Dataset

import torch 

import numpy as np

from torchvision import transforms


class Dataset_(Dataset):

	def __init__(self, noise=0.):
		self.noise=noise

	def __noise__(self, shape=None):

		if(shape is None):
			return torch.normal(mean=0.0, std=self.noise)

		return torch.normal(mean=0.0, std=torch.ones(shape)*self.noise)

	def sample(self,):
		pass
		
	def augment(self, image, mask = None, scale_ranges = [1, 1.1], img_size = [512, 512], translate = [0.1, 0.1], rotation = [-20, 20], brightness = None):
		tf_mask = None
		# Random rotation of -20 to 20 degress
		angle = transforms.RandomRotation.get_params(rotation)
		tf_image = transforms.functional.rotate(image, angle)
		#tf_image.show()
		if mask is not None:
			tf_mask = transforms.functional.rotate(mask, angle)

		# Random horizontal and vertical shift of -10% to 10%
		params = transforms.RandomAffine.get_params(degrees = [0, 0], translate = translate, \
		scale_ranges = [1, 1], img_size = img_size, shears = [0, 0])
		tf_image = transforms.functional.affine(tf_image, params[0], params[1], params[2], params[3])
		if mask is not None:
			tf_mask = transforms.functional.affine(tf_mask, params[0], params[1], params[2], params[3])

		# TODO: -10% to 10% may make more sense, due to the existance of images with black padding borders
		# Random zoom-in of 0% to 10%
		params = transforms.RandomAffine.get_params(degrees = [0, 0], translate = [0, 0], \
		scale_ranges = scale_ranges, img_size = img_size, shears = [0, 0])
		tf_image = transforms.functional.affine(tf_image, params[0], params[1], params[2], params[3])
		if mask is not None:
			tf_mask = transforms.functional.affine(tf_mask, params[0], params[1], params[2], params[3])

		# TODO: change brightness too
		# Random brightness change
		if brightness is not None:
			tf = transforms.ColorJitter(brightness = brightness)
			tf_image = tf(tf_image)

		if mask is not None:
			return (tf_image, tf_mask)
		else:
			return tf_image
