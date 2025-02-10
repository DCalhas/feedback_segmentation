import torch

import segmentation_models_pytorch as smp

from feedback_segmentation.layers import UnetFPI

from feedback_segmentation.layers.dirichlet import Dirichlet

from feedback_segmentation.layers.state import StateExpDecay

from feedback_segmentation.layers.equilibrium import EquilibriumModel

from feedback_segmentation.losses import evidential_loss

def entropy(x):
	"""
	Entropy from a concentration representation of the form (B, C, H, W) in the C dimension
	"""
	p=x/torch.sum(x, dim=1, keepdims=True)
	return -torch.sum(p*torch.log(p), dim=1, keepdims=True)

class NeumannFeedback(EquilibriumModel):
	"""
	This class implements a type of feedback that may incorporate a nonlinear function


	The input is $\mathbf{x} \in \mathbb{R}^{F}$ and we have a hidden state $\mathbf{h}(0) = \mathbf{0} \in \mathbb{R}^{F}$ 
	We have a system that evolves as 
	$\tilde{x}(t) = \left[\mathbf{x}(t) \cdot e^{\tau_1^{-1}\cdot t\cdot A}, \mathbf{h}(t) \cdot e^{\tau_2^{-1}\cdot t\cdot B}\right] \cdot e^{\tau_3^{-1} \cdot t \cdot C}$

	More specifically $\mathbf{h}(t) = F(\mathbf{x}(t))$
	
	"""
	UNetParams={"fb_filters": 2, "class_filters": 16, "registers": 0, "tau": 2e-1, "alpha": 1.5, "compute_error": True, "compute_softmax": True}
	
	@staticmethod
	def build_unet(in_filters, classes, shape, fb_filters=2, class_filters=16, registers=0, tau=2e-1, alpha=1.5, compute_error=True, compute_softmax=True, **kwargs):
		"""
		Returns a NeumannFeedback with a Unet as function
		"""
		F=UnetFPI(in_channels=in_filters, fb_filters=fb_filters, class_filters=class_filters, registers=registers, classes=classes)
		return NeumannFeedback(in_filters=in_filters, fb_filters=fb_filters, class_filters=class_filters, registers=registers, shape=shape, function=F, tau=tau, alpha=alpha, compute_error=compute_error, compute_softmax=compute_softmax)
		
	def __init__(self, in_filters, fb_filters, shape, function, class_filters=16, registers=0, tau=1., alpha=1.5, verbosity=False, **params):
		"""
		* in_filters: int
		* fb_filters: int
		* shape: tuple(int)
		* function: torch.nn.Module - should be able to receive an input of size (in_filters+fb_filters, *shape)
		* tau: float
		"""
		super(NeumannFeedback,self).__init__()

		self.in_filters=in_filters
		self.fb_filters=fb_filters
		self.class_filters=class_filters
		self.registers=registers

		self.h_class=StateExpDecay(filters=class_filters+fb_filters+registers, shape=shape, tau=1e3)

		self.function=function
		
		self.T=torch.nn.Parameter(torch.zeros(()), requires_grad=False)

		self.loss_fn=torch.nn.functional.cross_entropy
		
		self.ground_truth=None

		self.function.segmentation_head=smp.base.SegmentationHead(class_filters, function.classes)
		smp.base.initialization.initialize_head(self.function.segmentation_head)

		self.compute_softmax=params["compute_softmax"]
		
		self.compute_error=params['compute_error']
		if(self.compute_error):
			self.error_head=StateExpDecay(class_filters+fb_filters+registers, (512, 512), tau=tau)
			self.error_head.Q.requires_grad=True
			self.error_head.Qinv.requires_grad=True
			self.error_head.S.requires_grad=False
			torch.nn.init.xavier_uniform_(self.error_head.Q)
			torch.nn.init.xavier_uniform_(self.error_head.Qinv)

		self.layernorm_error=torch.nn.LayerNorm(shape,)
		self.layernorm_input=torch.nn.LayerNorm(shape,)

		self.verbosity=verbosity

		self.alpha=torch.nn.Parameter(torch.tensor(alpha), requires_grad=False)
		self.dt=torch.nn.Parameter(torch.tensor(1./5.), requires_grad=False)
		self.Tf=torch.nn.Parameter(torch.tensor(5.), requires_grad=False)

	def initT(self, T):
		torch.nn.init.constant_(self.dt, 1./(T))
		torch.nn.init.constant_(self.Tf, (T))

	@property
	def states(self):
		return {"h": self.h_class.state}

	def grad_step(self, x):
		"""
		This performs a step taking the autograd graph, unless specified in an outside scope as with torch.no_grad(): model.grad_step(x)
		"""
		h_normed=self.h_class.state
		
		ht=self.layernorm_input(h_normed[:,self.class_filters:self.class_filters+self.fb_filters])
		if(self.compute_softmax):
			ht=torch.nn.functional.softmax(ht, dim=1)
		
		xtilde=torch.concat((x, ht), dim=1)
		
		y=self.function(xtilde, T=0)
		
		y=self.layernorm_error(y)
		
		if(self.compute_error):
			y=self.error_head(y, T=self.T)

		error=y

		if(self.prev_training and self.verbosity): print(error.abs().sum(), end=" ")

		t=self.T
		self.h_class.state=h_normed+error*(self.Tf*self.dt-t*self.dt)**(self.alpha-1)/torch.exp(torch.special.gammaln(self.alpha))
		
		return self.prediction(), ht
		
	def step(self, x, enable_grad=True):
		"""
		This step accounts for steps that should not take gradients
		Each step in the NeumannFeedback computes the loss with respect to a target self.ground_truth which is given at training time
		"""
		if(not torch.is_nonzero(self.h_class.state.abs().sum())): self.h_class.state=self.h_class.state.to(x.device)
		
		if(self.prev_training and not torch.is_nonzero(self.h_class.state.abs().sum())): self.h_class.state=self.h_class.state.detach()
		
		if(enable_grad and self.prev_training):
			with torch.enable_grad():
				y_ht1, ht=self.grad_step(x)

				clf_loss=self.loss_fn(y_ht1[:,:self.function.classes].view(*self.ground_truth.shape[:2], -1), self.ground_truth.view(*self.ground_truth.shape[:2], -1), reduction="none")
				clf_loss.mean().backward(retain_graph=True)
		else:
			y_ht1, _=self.grad_step(x)
		
		torch.nn.init.constant_(self.T, self.T+1)

		return y_ht1[:,:self.function.classes]

	def prediction(self,):
		return self.function.segmentation_head(self.h_class.state[:,:self.class_filters])
		
	def forward(self, x, T=0, atol=1e-5):
		"""initialize hidden state with zeros, perform fixed point iteration and return the current state"""
		#initialize steps variable to 0
		torch.nn.init.zeros_(self.T)

		self.h_class.state=torch.zeros((x.shape[0], *self.h_class.state.shape[1:]), dtype=x.dtype, device=x.device)

		if(T==0): return self.prediction()

		if(T is not None):
			self.initT(T)
		
		super(NeumannFeedback,self).to_equilibrium(x, None, None, T=T-1, atol=atol)

		self.ground_truth=None

		return self.step(x, enable_grad=False)


class DirichletNeumannFeedback(NeumannFeedback, Dirichlet):
	"""
	This class implements a type of feedback that may incorporate a nonlinear function


	The input is $\mathbf{x} \in \mathbb{R}^{F}$ and we have a hidden state $\mathbf{h}(0) = \mathbf{0} \in \mathbb{R}^{F}$ 
	We have a system that evolves as 
	$\tilde{x}(t) = \left[\mathbf{x}(t) \cdot e^{\tau_1^{-1}\cdot t\cdot A}, \mathbf{h}(t) \cdot e^{\tau_2^{-1}\cdot t\cdot B}\right] \cdot e^{\tau_3^{-1} \cdot t \cdot C}$

	More specifically $\mathbf{h}(t) = F(\mathbf{x}(t))$
	
	"""
	UNetParams={"fb_filters": 2, "class_filters": 16, "registers": 0, "tau": 2e-1, "alpha": 1.5, "compute_entropy": True, "compute_error": True, "compute_attention": True, "temp_entropy": True, "compute_softmax": True}
	
	@staticmethod
	def build_unet(in_filters, classes, shape, fb_filters=2, class_filters=16, registers=0, tau=2e-1, alpha=1.5, compute_entropy=True, compute_error=True, compute_attention=True, temp_entropy=True, **kwargs):
		"""
		Returns a NeumannFeedback with a Unet as function
		"""
		F=UnetFPI(in_channels=in_filters, fb_filters=fb_filters, class_filters=class_filters, registers=registers, classes=classes)
		return DirichletNeumannFeedback(in_filters=in_filters, fb_filters=fb_filters, class_filters=class_filters, registers=registers, shape=shape, function=F, tau=tau, alpha=alpha, compute_entropy=compute_entropy, compute_error=compute_error, compute_attention=compute_attention, temp_entropy=temp_entropy, **kwargs)
		
	def __init__(self, **params):
		"""
		* in_filters: int
		* fb_filters: int
		* shape: tuple(int)
		* function: torch.nn.Module - should be able to receive an input of size (in_filters+fb_filters, *shape)
		* tau: float
		"""
		super(DirichletNeumannFeedback,self).__init__(**params)

		self.h_class=StateExpDecay(filters=params["function"].classes, shape=params["shape"], tau=1e3)
		
		self.function.segmentation_head=smp.base.SegmentationHead(params["fb_filters"]+params["class_filters"]+params["registers"], params["function"].classes)
		smp.base.initialization.initialize_head(self.function.segmentation_head)

		self.compute_entropy=params["compute_entropy"]
		self.compute_attention=params['compute_attention']
		if(self.compute_attention):
			raise NotImplementedError
		else:
			self.function.encoder._in_channels = params["in_filters"]+1#params["function"].classes
			smp.encoders._utils.patch_first_conv(model=self.function.encoder, in_channels=params["in_filters"]+1)#params["function"].classes)
		self.function.initialize()

		self.compute_error=params['compute_error']
		if(self.compute_error):
			self.error_head=StateExpDecay(params["function"].classes, params['shape'], tau=params["tau"])
			self.error_head.Q.requires_grad=True
			self.error_head.Qinv.requires_grad=True
			self.error_head.S.requires_grad=False

		self.loss_fn=evidential_loss

	def init_states(self,):
		#initialize beliefs
		for key in self.states.keys():
			torch.nn.init.ones_(self.states[key])
			
	def grad_step(self, x):
		"""
		This performs a step taking the autograd graph, unless specified in an outside scope as with torch.no_grad(): model.grad_step(x)
		"""
		h_normed=self.h_class.state

		ht=self.activation(h_normed)

		if(self.compute_entropy):
			ht=entropy(ht)
		
		xtilde=torch.concat((x, ht), dim=1)
		if(self.compute_attention):
			xtilde=self.attention(xtilde,)
		
		y=self.function(xtilde, T=0)
		y=self.function.segmentation_head(y)
		
		y=self.layernorm_error(y)
		if(self.compute_error):
			y=self.error_head(y, T=self.T)

		error=y
		
		if(self.prev_training and self.verbosity): print(error.abs().sum(), end=" ")
		
		t=self.T
		self.h_class.state=h_normed+error*(self.Tf*self.dt-t*self.dt)**(self.alpha-1)/torch.exp(torch.special.gammaln(self.alpha))
		
		return self.prediction(), ht

	def prediction(self,):
		return self.activation(self.h_class.state)
