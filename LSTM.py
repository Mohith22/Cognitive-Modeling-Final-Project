'''
Computational Cognitive Modeling - Prof. Brenden Lake and Prof. Todd Gureckis
Final Project - "Sequence Prediction using Bayesian Concept Learning"
'''

'''
LSTMS Models
'''

import numpy as np 
import math
import random

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable

np.set_printoptions(suppress=True)
np.random.seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.cuda.float if torch.cuda.is_available() else torch.float


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device, dtype=dtype)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device, dtype=dtype)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


def split_train_test(hypotheses, ratio):
	train_hypotheses = hypotheses[0:int(ratio*len(hypotheses))]
	test_hypotheses = hypotheses[int(ratio*len(hypotheses)):len(hypotheses)]
	return train_hypotheses, test_hypotheses

def read_file(filename):

	addition_hypotheses = open(filename).read()
	addition_hypotheses = addition_hypotheses.split("\n")
	addition_hypotheses_list = []
	for x in addition_hypotheses:
		trim_x = x[1:len(x)-1]
		trim_x = trim_x.replace(" ", "")
		trim_x_list = trim_x.split(",")
		addition_hypotheses_list.append(trim_x_list)
	return addition_hypotheses_list[:-1]

def data_XY(data):

	X = []
	Y = []
	for x in data:
		X.append(x[:-1])
		Y.append(x[-1])
	return X, Y

def data_prepare(addition_hypotheses):

	#print(len(addition_hypotheses))
	addition_hypotheses_train, addition_hypotheses_test = split_train_test(addition_hypotheses, 0.8)
	#print(len(addition_hypotheses_train))
	#print(len(addition_hypotheses_test))

	#print(len(multiplication_hypotheses))
	
	#multiplication_hypotheses_train, multiplication_hypotheses_test = split_train_test(multiplication_hypotheses, 0.8)
	#print(len(multiplication_hypotheses_train))
	#print(len(multiplication_hypotheses_test))

	#print(len(combination_hypotheses))
	#combination_hypotheses_train, combination_hypotheses_test = split_train_test(combination_hypotheses, 0.8)
	#print(len(combination_hypotheses_train))
	#print(len(combination_hypotheses_test))

	train_data = addition_hypotheses_train #+ addition_hypotheses_train + addition_hypotheses_train
	test_data = addition_hypotheses_test #+ addition_hypotheses_test + addition_hypotheses_test

	random.shuffle(train_data)
	random.shuffle(test_data)

	train_data_X, train_data_Y = data_XY(train_data)
	test_data_X, test_data_Y = data_XY(test_data)

	return train_data_X, train_data_Y, test_data_X, test_data_Y

def normalise(D):

	scaler = MinMaxScaler(feature_range=(0, 1))
	normalise_D = scaler.fit_transform(D)

	return normalise_D	

def inverse_normalise(D):

	scaler = MinMaxScaler(feature_range=(0, 1))
	invnormalise_D = scaler.inverse_transform(D)

	return invnormalise_D	

def one_hot_encode(x):

	v = [0 for i in range(100)]
	v[int(x)] = 1

	return v 

def one_hot_encode_data_train(D):

	d_one_hot = []
	for x in D:
		nx = []
		for i in x:
			nx.append(one_hot_encode(i))
		d_one_hot.append(nx)

	return np.array(d_one_hot)

def one_hot_encode_data_test(D):

	d_one_hot = []
	for x in D:
		d_one_hot.append(one_hot_encode(x))

	return np.array(d_one_hot)

def main():

	batch_size = 1

	addition_hypotheses = read_file("Sequences_Data/Addition_Hypotheses_Progressive2.txt")
	multiplication_hypotheses = read_file("Sequences_Data/Multiplication_Hypotheses_Progressive.txt")
	combination_hypotheses = read_file("Sequences_Data/Comb_Hypotheses_Progressive.txt")

	random.shuffle(addition_hypotheses)
	random.shuffle(multiplication_hypotheses)
	random.shuffle(combination_hypotheses)

	train_data_X, train_data_Y, test_data_X, test_data_Y = data_prepare(addition_hypotheses)

	train_data_X = one_hot_encode_data_train(train_data_X)
	train_data_Y = one_hot_encode_data_test(train_data_Y)
	
	test_data_X = one_hot_encode_data_train(test_data_X)
	test_data_Y = one_hot_encode_data_test(test_data_Y)

	featuresTrain = torch.from_numpy(train_data_X)
	targetsTrain = torch.from_numpy(train_data_Y).type(torch.LongTensor)
	
	featuresTest = torch.from_numpy(test_data_X)
	targetsTest = torch.from_numpy(test_data_Y).type(torch.LongTensor)

	# Pytorch train and test sets
	train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
	test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

	# data loader
	train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
	test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
	
	# Hyper-parameters
	sequence_length = 5
	input_size = 100
	hidden_size = 1
	num_layers = 1
	num_classes = 100
	num_epochs = 1
	learning_rate = 0.01

	model = RNN(input_size, hidden_size, num_layers, num_classes).to(device, dtype=dtype)

	# Loss and optimizer
	criterion = nn.MSELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	total_step = len(train_loader)

	print(len(train_loader))

	for epoch in range(num_epochs):
	    for i, (images, labels) in enumerate(train_loader):
	    	#print(images.shape)
	    	#print(labels.shape)
	    	images = images.reshape(-1, sequence_length, input_size).to(device, dtype=dtype)
	    	#print(images.shape)
	    	labels = labels.to(device, dtype=dtype)

	    	outputs = model(images)
	    	loss = criterion(outputs, labels)

	    	optimizer.zero_grad()
	    	loss.backward()
	    	optimizer.step()

	  
	    	print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

	
	with torch.no_grad():
	    correct = 0
	    total = 0
	    for images, labels in test_loader:
	        images = images.reshape(-1, sequence_length, input_size).to(device, dtype=dtype)
	        labels = labels.to(device, dtype=dtype)
	        outputs = model(images)
	        _, predicted = torch.max(outputs.data, 1)
	        predicted = predicted.to(device, dtype=dtype)
	        total += labels.size(0)
	        correct += (predicted == labels).sum().item()
	   		
	        print(predicted)
	        print(labels)
	    print(total)
	    print(correct)

	    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total)) 

if __name__== "__main__":
  main()







