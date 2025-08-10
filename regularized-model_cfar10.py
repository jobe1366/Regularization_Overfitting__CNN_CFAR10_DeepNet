
def get_model():
	
		class Regularized_Cifar10CnnModel(nn.Module):
			def __init__(self):
				super().__init__()
				self.network = nn.Sequential(
					
                nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),
				nn.ReLU(),
				nn.BatchNorm2d(16),
				nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
				nn.ReLU(),
				nn.BatchNorm2d(32), 
				nn.MaxPool2d(2,2),
				nn.Dropout(0.3),
					
				nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
				nn.ReLU(),
				nn.BatchNorm2d(64),
				nn.Conv2d(64,64,kernel_size=3,stride=1,padding=1),
				nn.ReLU(),
				nn.BatchNorm2d(64), 
				nn.MaxPool2d(2,2),
				nn.Dropout(0.3),
				
				nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
				nn.ReLU(),
				nn.BatchNorm2d(128),
				nn.Conv2d(128,128,kernel_size=3,stride=1,padding=1),
				nn.ReLU(),
				nn.BatchNorm2d(128), 
				nn.MaxPool2d(2,2),
				nn.Dropout(0.3),
				

				nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
				nn.ReLU(),
				nn.BatchNorm2d(256),
				nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1),
				nn.ReLU(),
				nn.BatchNorm2d(256), 
				nn.MaxPool2d(2,2), 
				nn.Dropout(0.3),
				
				nn.Flatten(),
				nn.Linear(256*2*2,512),
				nn.Dropout(0.3),
				nn.ReLU(),
				nn.BatchNorm1d(512),
				nn.Linear(512,64),
				nn.Dropout(0.3),
				nn.ReLU(),
				nn.Linear(64,10))
				nn.Softmax()
				

				
			def forward (self,x):
				return self.network(x)


		model = Regularized_Cifar10CnnModel().to(device)
		loss_fn = nn.CrossEntropyLoss()
		optimizer = Adam(model.parameters(), lr=1e-3)
		
		return model, loss_fn, optimizer



def train_batch(x, y, model, optimizer, loss_fn):
	
	model.train()
	prediction = model(x)
	batch_loss = loss_fn(prediction, y.type(torch.int64))
	optimizer.zero_grad()
	batch_loss.backward()
	optimizer.step()
	
	return batch_loss.item()




def get_data(batch_size = 500):
	 
	class CFAR_ten_Dataset(Dataset):
	
		def __init__(self, x, y):
			self.x = x
			self.y = y

		def __getitem__(self, ix):
			
			x, y = self.x[ix], self.y[ix]
			return torch.tensor(x/255).permute(2,0,1).to(device).float(), torch.tensor(y).float().to(device)
			
		def __len__(self):
			return len(self.x)
		
	train = CFAR_ten_Dataset(train_imgs, train_labels )
	trn_dl = DataLoader(train, batch_size=batch_size, shuffle=True)

	val = CFAR_ten_Dataset(validation_imgs, validation_labels)
	val_dl = DataLoader(val, batch_size=batch_size, shuffle=True)


	return trn_dl, val_dl




def val_loss(x, y, model,loss_fn ):
	 
	with torch.no_grad():
		prediction = model(x)
	val_loss = loss_fn(prediction, y.type(torch.int64))
	
	return val_loss.item()




def accuracy(x, y, model):

	model.eval()
	prediction = model(x)
	max_values, argmaxes = prediction.max(-1)
	is_correct = argmaxes == y

	return is_correct.cpu().numpy().tolist()




def main(num_epochs = 2):

	train_losses, train_accuracies = [], []
	val_losses, val_accuracies = [], []

	for epoch in range(num_epochs):
		train_epoch_losses, train_epoch_accuracies = [], []
		val_epoch_losses,val_epoch_accuracies = [], []

		# train loss
		for ix, (imgs, labels) in enumerate(trn_dl):
			batch_loss = train_batch(imgs, labels, model, optimizer, loss_fn)
			train_epoch_losses.append(batch_loss)
		train_epoch_loss = np.array(train_epoch_losses).mean()


		# validation loss
		for ix, (imgs, labels) in enumerate(val_dl):
			val_batch_loss = val_loss(imgs, labels, model,loss_fn )
			val_epoch_losses.append(val_batch_loss)
		val_epoch_losses = np.array(val_epoch_losses).mean()

		# train accuracy
		for ix, (imgs, labels) in enumerate(trn_dl):
			is_correct = accuracy(imgs, labels, model)
			train_epoch_accuracies.extend(is_correct)
		train_epoch_accuracy = np.mean(train_epoch_accuracies)

		# validation accuracy
		for ix, (imgs, labels) in enumerate(val_dl):
			val_is_correct = accuracy(imgs, labels, model)
			val_epoch_accuracies.extend(val_is_correct)
		val_epoch_accuracy = np.mean(val_epoch_accuracies)

		print(    f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
				f" | train_loss: {train_epoch_loss:.2f} | val_loss: {val_epoch_losses:.2f}"
				f" | train_accuracy: {train_epoch_accuracy:.2f} | val_accuracy: {val_epoch_accuracy:.2f}")

		train_losses.append(train_epoch_loss)
		train_accuracies.append(train_epoch_accuracy)
		val_accuracies.append(val_epoch_accuracy)
		val_losses.append(val_epoch_losses)

	return (train_losses, train_accuracies, val_losses, val_accuracies, num_epochs)
	 




def plotting_train_val_loss_accuracy_curves(train_losses, train_accuracies, val_losses, val_accuracies ,num_epochs):
	 
	epochs = np.arange(num_epochs)+1
	imgs_path = Path('/home/jobe/Desktop/my_venv/working/ML_proj/Regularization-methods-for-deep-Nets-main/IMGs')
	 
	plt.figure(figsize=(10,5))
	plt.plot(epochs, train_accuracies, 'bo', label='Training accuracy')
	plt.plot(epochs, val_accuracies, 'r', label='Validation accuracy')
	plt.title('Train/val accuracy for base_cnn_model_with_regularization')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.grid('off')
	plt.savefig(f'{imgs_path}/train_val_accuracy_with_regularization.png')

	plt.figure(figsize=(10,5))
	plt.plot(epochs, train_losses, 'bo', label='Training loss')
	plt.plot(epochs, val_losses, 'r', label='Validation loss')
	plt.title('Train/val loss for base_cnn_model_with_regularization')
	plt.xlabel('Epochs')
	plt.ylabel('Losses')
	plt.legend()
	plt.grid('off')
	plt.savefig(f'{imgs_path}/train_val_loss_with_regularization.png')
	print('______________PLOTTING_DONE_____________')
	 




def draw_confusion_matrix(network):

	confmx_path = Path('/home/jobe/Desktop/my_venv/working/ML_proj/Regularization-methods-for-deep-Nets-main/IMGs')
	label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] 
	cmap=plt.cm.Blues


	predicted_labels = []
	true_labels = []
	with torch.no_grad():
		for data in val_dl:
			images, labels = data
			y_pred = network(images)
			_, predicted_class = torch.max(y_pred, 1)
			predicted_labels.extend(predicted_class.cpu().numpy().tolist())
			true_labels.extend([int(item) for item in  labels.cpu().numpy()] )
	

	confmat = confusion_matrix(true_labels, predicted_labels)

	fig, ax = plt.subplots(figsize=(8 ,8))
	im = ax.imshow(confmat, interpolation='nearest', cmap=cmap)
	ax.figure.colorbar(im, ax=ax)

	ax.set(xticks=np.arange(confmat.shape[1]),
		   yticks=np.arange(confmat.shape[0]),
		   xticklabels=label_name, yticklabels=label_name,
		   title='Regularized Model Confusion Matrix',
		   ylabel='True label',
		   xlabel='Predicted label')
	
	plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

	fmt = 'd' 
	thresh = confmat.max() / 2.
	for i in range(confmat.shape[0]):
		for j in range(confmat.shape[1]):
			ax.text(j, i, format(confmat[i, j], fmt),
					ha="center", va="center",
					color="white" if confmat[i, j] > thresh else "black")
			
	fig.tight_layout()		 
	plt.xlabel('Predicted label') 
	plt.ylabel('True label')
	plt.savefig(f'{confmx_path}/Regularized_Class_Confusion_Matrix.png')
	print('\n\n_______________DONE_____________\n')






if __name__ == "__main__":
	
	import torch
	import torch.nn as nn
	import torch.nn.functional as F
	import torchvision.datasets as Datasets
	from torch.utils.data import DataLoader , Dataset
	from torch.optim import SGD, Adam
	import numpy as np
	from torchsummary import summary
	import matplotlib.pyplot as plt
	from pathlib import Path
	from sklearn.metrics import confusion_matrix

# select device for running model
	device = torch.device('cuda'  if torch.cuda.is_available()  else 'cpu')

# preparing data, model , loss_fn, optimizer to train base model

	trainset = Datasets.CIFAR10(root='./data', train=True, download=True)
	train_imgs = trainset.data
	train_labels = trainset.targets
	
	testset = Datasets.CIFAR10(root='./data', train=False, download=True)
	validation_imgs = testset.data
	validation_labels = testset.targets
	
	trn_dl, val_dl = get_data()
	model, loss_fn, optimizer = get_model()


# model summary

	print("\n\n ******REGULARIZED_MODEL_SUMMARY******\n\n")
	summary(model, torch.zeros(1,3,32,32))

# begin process

	print("\n\n ******TRANING_PROCESS******\n\n")
	train_losses, train_accuracies, val_losses, val_accuracies ,num_epochs = main(num_epochs=50)


# save model

	save_model_path = Path('/home/jobe/Desktop/my_venv/working/ML_proj/Regularization-methods-for-deep-Nets-main/save_models/Regularized_model_cifar10.pth')
	torch.save(model.state_dict(), save_model_path)

# plottig accuraccy and loss curves to evalute model performance(overfitting, under fitting, ... )
	print("\n\n ******PLOTTING CURVES******\n\n")
	plotting_train_val_loss_accuracy_curves(train_losses, train_accuracies, val_losses, val_accuracies ,num_epochs)

#plotting confusion matrix

	draw_confusion_matrix(model)