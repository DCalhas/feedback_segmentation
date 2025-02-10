import torch


class Dirichlet():

	def __init__(self,):
		pass

	def activation(self,x):
		return 1.+torch.nn.functional.softplus(x)