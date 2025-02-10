import torch

import numpy as np


class ExpDecay(torch.autograd.Function):


	def step(x, T, Delta):

		return torch.nn.functional.linear(x.permute(0,2,3,1), torch.diag(torch.exp(T*Delta))).permute(0,3,1,2)

	def backwardH(grad, Delta):
		return torch.nn.functional.linear(grad.permute(0,2,3,1), (torch.diag(torch.exp(Delta))).T).permute(0,3,1,2)

	def dDelta(Delta):
		derivative=(Delta>0).to(Delta.device, Delta.dtype)
		zeros=(Delta!=0).to(Delta.device, Delta.dtype)
		return (derivative*2-1.)*zeros#scale and shift

	
	def backwardDelta(grad, x, Delta):
		return (grad.permute(0,2,3,1)*x.permute(0,2,3,1)*torch.exp(Delta)).mean((0,1,2,))
	
	@staticmethod
	def forward(ctx, x, T, Delta):
		
		z = ExpDecay.step(x, T, Delta)
		
		ctx.save_for_backward(x, T, Delta, z)
		
		return z

	@staticmethod
	def backward(ctx, grad_output):
		
		x, T, Delta, z = ctx.saved_tensors

		gradX=gradT=gradDelta=None

		neumann_vH=grad_output.clone()
		neumann_gH=grad_output.clone()

		for i in range(int(T.item())):
			neumann_vH=ExpDecay.backwardH(neumann_vH, Delta)

			neumann_gH+=neumann_vH

		gradX=ExpDecay.backwardH(neumann_gH, Delta)
		gradDelta=ExpDecay.backwardDelta(neumann_gH, x, Delta)

		return gradX, gradT, gradDelta


class StateExpDecay(torch.nn.Module):
	"""
	Example usage:
		>>> layer=StateExpDecay(64, (64, 512, 512,), tau=0.5)
		>>> x=x.to(device="cuda:0")
		>>> layer.to(device="cuda:0")
		>>> z=layer(x)
		>>> z.sum().backward()
		>>> 
		>>> print(z.shape)
	"""

	def __init__(self, filters, shape, tau=0.1, alpha=1.):
		super(StateExpDecay, self).__init__()

		self.filters=filters

		self.tau=torch.nn.Parameter(tau*torch.ones((),), requires_grad=False)
		self.T=torch.nn.Parameter(torch.ones((),), requires_grad=False)
		self.alpha=torch.nn.Parameter(alpha*torch.ones((),), requires_grad=False)
		self.state=torch.zeros((1,filters, *shape), requires_grad=False)

		#init eigenvalues
		self.W=torch.nn.Parameter(torch.ones(filters,filters), requires_grad=False)
		torch.nn.init.xavier_uniform_(self.W)
		
		#perform eigendecomposition
		S, Q=torch.linalg.eig(self.W@self.W.T)#symmetric matrix
		self.S=torch.nn.Parameter(S.real.to(dtype=torch.float32), requires_grad=True)
		self.Q=torch.nn.Parameter(Q.real.to(dtype=torch.float32), requires_grad=False)
		self.Qinv=torch.nn.Parameter(torch.linalg.inv(Q.real).to(dtype=torch.float32), requires_grad=False)

		torch.nn.init.ones_(self.S)
		
		self.autograd_fn=ExpDecay

	def solve(self, x, S, Q, Qinv, T,):
		
		z=torch.nn.functional.linear(x.permute(0,2,3,1), Q).permute(0,3,1,2,)
		
		z2=self.autograd_fn.apply(z, self.T*T, S)
		
		return torch.nn.functional.linear(z2.permute(0,2,3,1), Qinv).permute(0,3,1,2,)

	def forward(self, x, T=5):
		
		return self.solve(x, (1/self.tau)*-1.*torch.abs(self.S)**self.alpha, self.Q, self.Qinv, T)


class StateExpDecayRandom(StateExpDecay):


	def __init__(self, *args, **kwargs):
		super(StateExpDecayRandom, self).__init__(*args, **kwargs)

		self.mask=torch.nn.Parameter(torch.bernoulli(0.5*torch.ones((self.filters,))), requires_grad=False)

		#set eigenvectors trainable
		self.Q.requires_grad=True
		self.Qinv.requires_grad=True

	def forward(self, x, T=5):
		return super(StateExpDecayRandom, self).solve(x, self.mask*(1/self.tau)*-1.*torch.abs(self.S)**self.alpha, self.Q, self.Qinv, T)



class GradientDescent(torch.autograd.Function):


	def step(x, h, tau):

		return h+tau*(x-h)

	def backwardX(grad, x, h, tau):

		return grad*(tau)

	def backwardH(grad, x, h, tau):

		return grad*(1-tau)
	
	@staticmethod
	def forward(ctx, x, h, tau):
		
		z = GradientDescent.step(x, h, tau)

		ctx.save_for_backward(x, h , tau, z)
		
		return z

	@staticmethod
	def backward(ctx, grad_output):
		x, h , tau, z = ctx.saved_tensors

		gradX=gradH=gradTau=None

		neumann_vH=grad_output.clone()
		neumann_gH=grad_output.clone()

		for i in range(100):			
			neumann_vH=GradientDescent.backwardH(neumann_vH, x, h, tau)
			
			neumann_gH+=neumann_vH

		gradX=GradientDescent.backwardX(neumann_gH, x, h, tau)
		gradH=GradientDescent.backwardH(neumann_gH, x, h, tau)
		
		return gradX, gradH, gradTau




class StateGradient(torch.nn.Module):
	"""
	Example usage:
		>>> layer=StateGradient(64, (64, 512, 512,), GradientDescent)
		>>> x=x.to(device="cuda:0")
		>>> layer.to(device="cuda:0")
		>>> z=layer(x)
		>>> z.sum().backward()
		>>> 
		>>> print(z.shape)
	"""

	def __init__(self, filters, shape, tau=0.1):
		super(StateGradient, self).__init__()

		shape=(filters,)+shape
		flattened_shape=np.prod(shape)

		self.filters=filters

		self.h0=torch.nn.Parameter(torch.zeros((1,flattened_shape),), requires_grad=False)
		self.tau=torch.nn.Parameter(tau*torch.ones((),), requires_grad=False)
		self.state=torch.zeros((1,flattened_shape), requires_grad=False)

		self.autograd_fn=GradientDescent

	def to(self, device="cuda:0", **kwargs):
		super(StateGradient, self).to(device=device, **kwargs)

		self.h0.to(device=device)
		self.state.to(device=device)
		self.tau.to(device=device)
		
	def forward(self, x):

		#flatten input
		shape=x.shape
		x=x.reshape((x.shape[0],-1))
		
		#wrap this in a torch autograd function
		if(torch.is_nonzero(self.state.abs().sum())):
			self.state=self.autograd_fn.apply(x, self.state, self.tau)
			
		else:
			self.state=self.autograd_fn.apply(x, self.h0, self.tau)

		return self.state.reshape(shape)