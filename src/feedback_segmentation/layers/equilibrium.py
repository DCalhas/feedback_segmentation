import torch

import numpy as np

class EquilibriumModel(torch.nn.Module):
	"""
	This class implements fixed point iteration
	"""
	def init_states(self,):
		#initialize beliefs
		for key in self.states.keys():
			torch.nn.init.zeros_(self.states[key])
	
	@property
	def states(self,):
		"""
		This method should be implemneted by the subclass
		"""
		raise NotImplementedError

	def step(self, x):
		"""
		This method should be implemented by the subclass

		A step in differs from system to system
		"""
		raise NotImplementedError

	def prediction(self,):
		"""
		This method should be implemented by the subclass

		This returns the prediction given current state of the model
		"""
		raise NotImplementedError

	def to_equilibrium(self, ims, masks, loss_fn, T=None, atol=1e-2):
		"""
		Fixed point iteration to equilibrium state
		"""
		self.prev_training=self.training
		self.train(False)
		self.init_states()

		if(T<=0):
			self.train(self.prev_training) 
			return

		prev=np.inf

		if(T is None):
			T=2**32-1
		
		for t in range(T):
			out=self.step(ims,)
			
			eq=0.
			for key in self.states.keys():
				eq+=(self.states[key].to(ims.device)).abs().sum()
			
			if(torch.abs(eq-prev) <= atol):
				self.train(self.prev_training)
				return
			prev=eq
		self.train(self.prev_training)

	def equilibrium_nograph(self, x, T=None, atol=1e-2):
		"""Sequentially pass `x` trough model`s encoder, decoder and heads"""		
		
		#first go to equilibrium without computing the autograd graph
		with torch.no_grad():
			self.to_equilibrium(x, None, None, T=T-1, atol=atol)