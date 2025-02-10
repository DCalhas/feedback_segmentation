import torch

class Loss(torch.nn.Module):
	"""
	Standard class for a loss class in torch

	This only specifies the type of reduction of the child class
	"""

	def __init__(self, reduction="mean"):

		super(Loss, self,).__init__()

		self.reduction=reduction

	def forward(self, x):

		assert len(x.shape)<=1, "E: Rank of tensor is "+str(len(x.shape))+" when it should have rank 1."

		if(self.reduction=="none"):
			return x
		elif(self.reduction=="mean"):
			return torch.mean(x)
		elif(self.reduction=="sum"):
			return torch.sum(x)
		else:
			raise NotImplementedError

			
class CombinedLoss(Loss):
	"""
	This classes combines a number of loss functions
	"""

	def __init__(self, losses, weights=None, **kwargs):

		"""
		Inputs:
			* tuple(object): tuple with classes specifying the losses
			* list(float32): weights of each loss
		"""

		super(CombinedLoss, self).__init__(**kwargs)

		self.losses=losses
		self.weights=weights
		if(weights is None):
			self.weights=torch.ones((len(losses),1), dtype=torch.float32)
			self.weights/=self.weights.shape[0]

		#force reduction mode to be none
		#for loss_fn in self.losses:
		#	loss_fn.reduction="none"

	def forward(self, y_pred, y_true):

		loss=0.0

		self.weights=self.weights.to(device=y_pred.device)

		for i in range(len(self.losses)):
			loss+=self.weights[i]*self.losses[i](y_pred, y_true)

		return super(CombinedLoss, self).forward(loss)