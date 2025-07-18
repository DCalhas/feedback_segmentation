import feedback_segmentation as fbs

import torch

## State the number of classes you want, the resolution of the images
n_classes=2
resolution=(512,512)
noise=10.0
n_samples=10
device="cuda:0"
epochs=10

# Create a dataset
dataset=fbs.data.PolygonsDataset(n_samples,classes=n_classes-1,noise=noise,resolution=resolution)
dataset.sample()

# Create the model
x,_=dataset.__getitem__(0)
model=fbs.layers.NeumannFeedback.build_unet(x.shape[0], n_classes, x.shape[1:], **fbs.layers.NeumannFeedback.UNetParams)

# Train the model
loss_fn=torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
train_dataset_loader=torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
model.to(device)
for epoch in range(epochs):
	model.train()
	epoch_loss=0.
	n_batches=0
	for batch in train_dataset_loader:
		imgs = batch[0]
		true_masks = batch[1]
		imgs = imgs.to(device=device,)
		true_masks = true_masks.to(device=device,)
		#the forward call of the NeumannFeedback model performs backward operations
		optimizer.zero_grad()
		model.ground_truth=true_masks
		loss=loss_fn(model(imgs, T=5), true_masks)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
		n_batches+=1
		if(n_batches%16==0):
			print("Loss "+str(loss.item()), end="\r")
	print("Epoch "+str(epoch+1)+" with loss "+str(epoch_loss/n_batches), end="\n")


# Evaluate the model in a different dataset
with torch.no_grad():
	model.eval()
	test_dataset=fbs.data.PolygonsDataset(200,classes=n_classes-1,noise=noise,resolution=resolution)
	test_dataset.sample()
	test_dataset_loader=train_dataset_loader=torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)
	print(fbs.metrics.IoU(test_dataset_loader, model, T=10, device=device))
