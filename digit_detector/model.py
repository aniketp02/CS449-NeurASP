import torch
import torchvision
import cnn_model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import random
import numpy as np
from dataLoader import Loader
import os
import cv2
import matplotlib as plt


torch.backends.cudnn.enabled = False
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class Trainer:
	def __init__(self):

		self.batch_size = 32  # Batch Size
		self.num_epochs = 20  # Number of Epochs to train for
		self.lr = 0.001       # Learning rate
		self.momentum =  0.5
		
		self.model = cnn_model.Net()
		self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
		

	def load_data(self):
		# Initializing the dataloaders
		self.train_loader = torch.utils.data.DataLoader(
			torchvision.datasets.MNIST('MNIST/raw/train-images-idx3-ubyte', train=True, download=True,
			transform=torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize((0.1307,), (0.3081))
				# The nos. are global mean and global std for MNIST dataset
			])),
			batch_size=self.batch_size,
			shuffle=True
		)

		self.test_loader = torch.utils.data.DataLoader(
			torchvision.datasets.MNIST('MNIST/raw/t10k-images-idx3-ubyte', train=False, download=True,
			transform=torchvision.transforms.Compose([
				torchvision.transforms.ToTensor(),
				torchvision.transforms.Normalize((0.1307,), (0.3081))
				# The nos. are global mean and global std for MNIST dataset
			])),
			batch_size=self.batch_size,
			shuffle=True
		)


	def save_model(self):
		torch.save(self.model.state_dict(), f'assets/model.pth')
		torch.save(self.optimizer.state_dict(), f'assets/optimizer.pth')


	def load_model(self):
		if os.path.exists('assets/model.pth'):
			self.model.load_state_dict(torch.load('assets/model.pth'))
			self.model.eval()
			pass
		else:
			raise Exception('Model not trained')

	def train(self):
		if not self.model:
			return

		print("Training...")
		train_loss_values = []
		for epoch in range(self.num_epochs):
			train_loss = self.run_epoch(epoch)
			train_loss_values.append(train_loss)

			self.save_model()

			print(f'	Epoch #{epoch+1} trained')
			print(f'		Train loss: {train_loss:.3f}')
		print('Training Complete')

	def test(self):
		if not self.model:
			return 0

		print(f'Running test...')
		# Initialize running loss
		running_loss = 0.0
		test_loss_values = []


		i = 0  # Number of batches
		correct = 0  # Number of correct predictions
		with torch.no_grad():
			loss_test = 0
			for data, target in self.test_loader:
				# Find the predictions
				preds = self.model(data)

				# Find the loss
				loss_test += F.nll_loss(preds, target, size_average=False).item()

				pred = preds.data.max(1, keepdim=True)[1]
				correct += pred.eq(target.data.view_as(pred)).sum()

			# Update running_loss
			running_loss = float(loss_test)
			test_loss_values.append(running_loss)
			i += 1
		
		print(f'	Test loss: {(running_loss):.3f}')
		print(f'	Test accuracy: {(correct/data.shape[0]):.2f}%')

		return correct/data.shape[0]

	def run_epoch(self, epoch):
		# Initialize running loss
		running_loss = 0.0

		self.model.train()

		i = 0 # Number of batches
		for batch_idx, (data, target) in enumerate(self.train_loader):
			self.optimizer.zero_grad()

			# data = data / 255
			output = self.model(data)

			self.loss = F.nll_loss(output, target)
			self.loss.backward()
			self.optimizer.step()
			
			if(batch_idx % 100 == 0):
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                	epoch, batch_idx * len(data), len(self.train_loader.dataset),
                	100. * batch_idx / len(self.train_loader), self.loss.item()))
			running_loss = float(self.loss.item())

			i += 1
		
		return running_loss

	def predict(self, image):
		prediction = 0
		if not self.model:
			return prediction

		pred_img = cv2.imread(image, 0)
		pred_img = pred_img / 255
		print(pred_img.shape)

		pred_img = cv2.resize(pred_img, (28, 28))
		pred_img = torch.Tensor(pred_img)
		re_pred_img = torch.reshape(pred_img, shape=(1, 28, 28))
		print(re_pred_img.shape)

		# Predict the digit value using the model
		prediction_arr = self.model(re_pred_img).cpu().detach().numpy()
		prediction = np.argmax(prediction_arr)

		# End Editing
		return prediction


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Model Trainer')
	parser.add_argument('-train', action='store_true', help='Train the model')
	parser.add_argument('-test', action='store_true', help='Test the trained model')
	parser.add_argument('-path', help='Path of the image for prediction!')
	parser.add_argument('-predict', action='store_true', help='Make a prediction on a randomly selected test image')

	options = parser.parse_args()

	t = Trainer()
	if options.train:
		t.load_data()
		t.train()
		t.test()
	if options.test:
		t.load_data()
		t.load_model()
		t.test()
	if options.predict:
		t.load_data()
		try:
			t.load_model()
		except:
			print("Model not loaded! Configure the model path properly!")

		val = t.predict(options.path)
		print(f'Predicted: {val}')

		# for id, (data, target) in enumerate(t.test_loader):
		# 	i = np.random.randint(0,data.shape[0])
		# 	print("input shape", data[i].shape)
		# 	val = t.predict(data[i])
		# 	print(f'Predicted: {val}')
		# 	print(f'Actual: {target[i]}')

			# image = data[i].cpu().detach().numpy()
			# image = image.reshape((28,28))
			# image = cv2.resize(image, (0,0), fx=16, fy=16)
			# cv2.imwrite('out_{}_{}.jpg'.format(id, val), image * 255)
