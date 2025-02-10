import torch

from feedback_segmentation.layers.dirichlet import Dirichlet

import numpy as np

def compute_IoU_precision(a, b, c):
	"""
	Inputs:
		* np.ndarray: B, C, H, W shape
		* np.ndarray: B, C, H, W shape
	"""
	assert a.shape[0]==1 and b.shape[0]==1
	
	_c=np.eye(a.shape[1])[c].reshape((1,-1,1,1))
	indices=np.where(np.all(b==_c, axis=1) & np.all(a==_c, axis=1))
	
	return (len(indices[0]))/(np.sum(a[:,c])+1e-3)


def compute_IoU_recall(a, b, c):
	"""
	Inputs:
		* np.ndarray: B, C, H, W shape
		* np.ndarray: B, C, H, W shape
	"""
	assert a.shape[0]==1 and b.shape[0]==1
	
	_c=np.eye(a.shape[1])[c].reshape((1,-1,1,1))
	indices=np.where(np.all(b==_c, axis=1) & np.all(a==_c, axis=1))
	
	return (len(indices[0]))/(np.sum(b[:,c])+1e-3)


def compute_IoU(a, b):
	"""
	Inputs:
		* np.ndarray: B, C, H, W shape
		* np.ndarray: B, C, H, W shape
	"""
	assert a.shape[0]==1 and b.shape[0]==1
	
	intersection=np.sum(np.logical_and.reduce(a==b, axis=1).astype(np.float32), axis=(1,2))
	union=np.sum(np.logical_or.reduce(a+b>0., axis=1).astype(np.float32), axis=(1,2))

	return np.mean((intersection)/(union+1), axis=0)


def IoU(loader, model, device='cuda:0', T=0):
	

	n_samples=0
	for _, x in enumerate(loader):
		n_samples+=1
		n_classes=x[1].shape[1]
	
	iou=np.empty((n_samples, n_classes*2+1), dtype=np.float32)#first position is for global IoU the rest is precision
	
	for i, x in enumerate(loader):
		print("I: On instance "+str(i)+"/"+str(n_samples), end="\r")
	
		image, mask=x
		if(T==0):
			outputs=model(image.to(device=device,))
		else:
			outputs=model(image.to(device=device,), T=T)
			
		if(isinstance(model,Dirichlet)):
			outputs=torch.distributions.Dirichlet(outputs.permute((0,2,3,1))).sample().permute((0,3,1,2))
		else:
			outputs=torch.nn.functional.softmax(outputs, dim=1)
		outputs=(outputs>(1/outputs.shape[1])).to('cpu:0', dtype=torch.float32)
	
		iou[i, 0]=compute_IoU(outputs.numpy(), mask.cpu().numpy())
		for c in range(n_classes):
			iou[i, c+1]=compute_IoU_recall(outputs.numpy(), mask.cpu().numpy(), c=c)
			iou[i, c+1+n_classes]=compute_IoU_precision(outputs.numpy(), mask.cpu().numpy(), c=c)

	print("I: Finished metric computation successfully", end="\n")
	return np.mean(iou, axis=0), np.std(iou, axis=0)
