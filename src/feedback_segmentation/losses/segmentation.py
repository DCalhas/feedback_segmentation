import torch

from feedback_segmentation.losses import Loss


def dice_loss(input, target, use_weights = False, k = 0, eps = 0.0001):
	"""
	Returns the Generalized Dice Loss Coefficient of a batch associated to the input and target tensors. In case `use_weights` \
		is specified and is `True`, then the computation of the loss takes the class weights into account.

	Args:
		input (torch.FloatTensor): NCHW tensor containing the probabilities predicted for each class.
		target (torch.LongTensor): NCHW one-hot encoded tensor, containing the ground truth segmentation mask. 
		use_weights (bool): specifies whether to use class weights in the computation or not.
		k (int): weight for pGD function. Default is 0 for ordinary dice loss.
	"""
	if input.is_cuda:
		s = torch.FloatTensor(1).cuda(input.device).zero_()
	else:
		s = torch.FloatTensor(1).zero_()

	class_weights = None
	for i, c in enumerate(zip(input, target)):
		if use_weights:
			class_weights = torch.pow(torch.sum(c[1], (1,2)) + eps, -2)
		s = s + __dice_loss(c[0], c[1], class_weights, k=k)

	return s / (i + 1)

def __dice_loss(input, target, weights = None, k = 0, eps = 0.0001):
	"""
	Returns the Generalized Dice Loss Coefficient associated to the input and target tensors, as well as to the input weights,\
	in case they are specified.

	Args:
		input (torch.FloatTensor): CHW tensor containing the classes predicted for each pixel.
		target (torch.LongTensor): CHW one-hot encoded tensor, containing the ground truth segmentation mask. 
		weights (torch.FloatTensor): 2D tensor of size C, containing the weight of each class, if specified.
		k (int): weight for pGD function. Default is 0 for ordinary dice loss.
	"""  
	n_classes = input.size()[0]

	if weights is not None:
		for c in range(n_classes):
			intersection = (input[c] * target[c] * weights[c]).sum()
			union = (weights[c] * (input[c] + target[c])).sum() + eps
	else:
		intersection = torch.dot(input.view(-1), target.view(-1))
		union = torch.sum(input) + torch.sum(target) + eps	

	gd = (2 * intersection.float() + eps) / union.float()
	return 1 - (gd / (1 + k*(1-gd)))

class GeneralizedDiceLoss(Loss):
	"""
	GLD loss implemented as done in https://arxiv.org/abs/2106.11447

	"""

	def __init__(self, n_classes, kappa=0.75, **kwargs):

		super(GeneralizedDiceLoss, self,).__init__(**kwargs)
		
		self.n_classes=n_classes
		self.kappa=kappa
		self.softmax=torch.nn.Softmax(dim=1)

	def __call__(self, y_pred, y_true):

		y_pred=self.softmax(y_pred)

		return dice_loss(y_pred, y_true, k=self.kappa)

		class_weights=1/(torch.sum(y_true, dim=(2, 3)))**2

		gdl=1-2*torch.sum(class_weights*torch.sum(y_true*y_pred, dim=(2,3)), dim=1)/torch.sum(class_weights*torch.sum(y_true+y_pred, dim=(2,3)), dim=1)

		loss=gdl/(1+self.kappa*(1-gdl))

		return super(GeneralizedDiceLoss, self).forward(loss)


class FocalLoss(Loss):
	"""
	Focal loss implemented as done in https://arxiv.org/abs/2106.11447
	
	Gives higher penalties to pixels that were harder to classify
	"""

	def __init__(self, alpha=0.25, gamma=2., **kwargs):

		super(FocalLoss, self).__init__(**kwargs)

		self.alpha=alpha
		self.gamma=gamma

	def __call__(self, y_pred, y_true):
	
		# compute softmax over the classes axis
		log_pred_soft = y_pred.log_softmax(dim=1)
	
		# compute the actual focal loss
		loss_tmp = -torch.pow(1.0 - log_pred_soft.exp(), self.gamma) * log_pred_soft * y_true
	
		broadcast_dims = [-1] + [1] * len(y_pred.shape[2:])
		
		alpha_fac = torch.tensor([1 - self.alpha] + [self.alpha] * (y_true.shape[1] - 1), dtype=loss_tmp.dtype, device=loss_tmp.device)
		alpha_fac = alpha_fac.view(broadcast_dims)
		loss_tmp = alpha_fac * loss_tmp
		loss_tmp = torch.mean(loss_tmp, dim=(1,2,3))

		return super(FocalLoss, self).forward(loss_tmp)
