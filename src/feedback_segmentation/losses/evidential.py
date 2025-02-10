import torch


def kl(alpha):
	num_classes=alpha.shape[1]
	beta = torch.ones((1, num_classes, 1, 1), dtype=torch.float, device=alpha.device)
	S_alpha = torch.sum(alpha, dim=1, keepdim=True)
	S_beta = torch.sum(beta, dim=1, keepdims=True)
	lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
	lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)

	dg0 = torch.digamma(S_alpha)
	dg1 = torch.digamma(alpha)

	return torch.sum((alpha-beta)*(dg1-dg0), dim=1, keepdim=True) + lnB + lnB_uni

def evidential_loss(alpha, y, reduction=None):
	"""
	Arguments:
	y : Tensor of ground-truth labels (batch_size, num_classes, d1, ..., dk).
	alpha : Tensor of Dirichlet concentration parameters (batch_size, num_classes, d1, ..., dk).
	
	Returns:
	Total evidential loss.

	This code was sourced from the "Computing a human-like reaction time metric from stable recurrent vision models" (https://arxiv.org/abs/2306.11582)
	"""
	y = y.float()
	# S is the sum of alphas across the classes
	
	S = torch.sum(alpha, dim=1, keepdim=True)
	m = alpha / S

	A = torch.sum((y-m)**2, dim=1, keepdim=True)
	B = torch.sum(alpha*(S-alpha)/(S*S*(S+1)), dim=1, keepdim=True)

	# Calculate the risk loss
	L_risk = A+B

	# Calculate the balance loss (KL divergence with the uniform Dirichlet prior)

	L_bal = kl((alpha-1)*(1-y) +1)

	# Total loss is the sum of risk and balance losses
	loss = L_risk.mean() + L_bal.mean()
	return loss