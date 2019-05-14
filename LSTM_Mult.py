import numpy as np 
import random

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
        self.softmax = nn.Softmax()
    
    def forward(self, x):
        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device, dtype=dtype)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device, dtype=dtype)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out

def reverse(x):

	numbers = [0,0,0,0,0]
	i = 0
	while x:
		numbers[i] = x % 10
		x=x//10
		i+=1
	return numbers

def encode_number(x):

	v1 = [0 for i in range(10)]
	v2 = [0 for i in range(10)]
	v3 = [0 for i in range(10)]
	v4 = [0 for i in range(10)]
	v5 = [0 for i in range(10)]

	number_list = reverse(x)

	v5[number_list[0]] = 1
	v4[number_list[1]] = 1
	v3[number_list[2]] = 1
	v2[number_list[3]] = 1
	v1[number_list[4]] = 1

	return v1 + v2 + v3 + v4 + v5

def data_point(c, d, length):

	data = []
	while (length):
		if c > 999:
			return []
		data.append(c)
		c = c + d
		length-=1
	return data 

def encode_X(data_set_x):

	data_set_x_encoded = []
	for x in data_set_x:
		interm = []
		for i in x:
			interm.append(encode_number(i))
		data_set_x_encoded.append(interm)
	return data_set_x_encoded

def encode_Y(data_set_y):

	data_set_y_encoded = []
	for x in data_set_y:
		data_set_y_encoded.append(encode_number(x))
	return data_set_y_encoded

def train_test_split(data_set_x, data_set_y):
	p = 0.8
	data_set_x_train = data_set_x[0:int(p * len(data_set_x))]
	data_set_x_test = data_set_x[int(p * len(data_set_x)):len(data_set_x)]

	data_set_y_train = data_set_y[0:int(p * len(data_set_y))]
	data_set_y_test = data_set_y[int(p * len(data_set_y)):len(data_set_y)]

	return np.array(data_set_x_train), np.array(data_set_y_train), np.array(data_set_x_test), np.array(data_set_y_test)

def gen_data(d, length):

	data_set_x = []
	data_set_y = []
	data_set = []
	for c in range(1, 1000):
		if (len(data_point(c, d, length))):
			data_set.append(data_point(c,d,length))
	random.shuffle(data_set)
	return data_set

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

def main():

	sequence_length = 5
	input_size = 50
	hidden_size = 50
	num_layers = 2
	num_classes = 50
	num_epochs = 60
	learning_rate = 0.01
	batch_size = 16
	
	data_set = []
	data_set_x = []
	data_set_y = []
	'''for i in range(2,11):
		data= gen_data(i,sequence_length+1)
		data_set += data

	random.shuffle(data_set)'''

	addition_hypotheses = read_file("Multiplication_Hypotheses_Progressive_LSTM.txt")
	random.shuffle(addition_hypotheses)

	data_set = addition_hypotheses

	data_set_x = []
	data_set_y = []

	for x in range(len(data_set)):
		data_set_x.append(list(map(int, data_set[x][0:-1])))
		data_set_y.append(int(data_set[x][-1]))


	print(len(data_set_x))
	print(len(data_set_y))


	for x in range(len(data_set_x)):
		print(data_set_x[x], data_set_y[x])

	'''data_set_x.append([1,2,4,8,18])
	data_set_y.append(36)

	data_set_x.append([1,3,9,27,54])
	data_set_y.append(162)'''

	data_set_x = encode_X(data_set_x)
	data_set_y = encode_Y(data_set_y)


	data_set_x_train, data_set_y_train, data_set_x_test, data_set_y_test = train_test_split(data_set_x, data_set_y)

	print(data_set_x_train.shape)
	print(data_set_x_test.shape)

	featuresTrain = torch.from_numpy(data_set_x_train)
	targetsTrain = torch.from_numpy(data_set_y_train).type(torch.LongTensor)
	
	featuresTest = torch.from_numpy(data_set_x_test)
	targetsTest = torch.from_numpy(data_set_y_test).type(torch.LongTensor)

	# Pytorch train and test sets
	train = torch.utils.data.TensorDataset(featuresTrain,targetsTrain)
	test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

	# data loader
	train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
	test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)
	

	model = RNN(input_size, hidden_size, num_layers, num_classes).to(device, dtype=dtype)

	# Loss and optimizer
	criterion = nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

	total_step = len(train_loader)

	#print(len(train_loader))
	
	for epoch in range(num_epochs):
	    for i, (images, labels) in enumerate(train_loader):
	    	#print(images.shape)
	    	#print(labels.shape)
	    	images = images.reshape(-1, sequence_length, input_size).to(device, dtype=dtype)
	    	#print(images.shape)
	    	labels = labels.to(device, dtype=dtype)
	    	#print(images.shape)
	    	outputs = model(images)
	    	#print(outputs)
	    	#print(labels)
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
	        predicted = outputs #torch(outputs.data, 1)
	        predicted = predicted.to(device, dtype=dtype)
	        total += labels.size(0)
	        predicted = predicted.detach().numpy()
	        labels = labels.detach().numpy()

	        predicted_new = np.zeros_like(predicted)

	        predicted = predicted.tolist()
	        labels = labels.tolist()
	        predicted_new = predicted_new.tolist()

	        #print(predicted_new)

	        for i in range(len(predicted)):
	        	pred1 = predicted[i][0:10]
	        	pred2 = predicted[i][10:20]
	        	pred3 = predicted[i][20:30]
	        	pred4 = predicted[i][30:40]
	        	pred5 = predicted[i][40:50]
	        	index1 = predicted[i].index(max(pred1))
	        	index2 = predicted[i].index(max(pred2))
	        	index3 = predicted[i].index(max(pred3))
	        	index4 = predicted[i].index(max(pred4))
	        	index5 = predicted[i].index(max(pred5))
	        	predicted_new[i][index1] = 1 
	        	predicted_new[i][index2] = 1 
	        	predicted_new[i][index3] = 1 
	        	predicted_new[i][index4] = 1 
	        	predicted_new[i][index5] = 1 
	        for i in range(len(predicted_new)):

	        	if (predicted_new[i] == labels[i]):
	        		correct+=1
	        #print(predicted_new)
	        #print(labels)

	        #correct += (predicted == labels).sum().item()
	   	
	    #print(total)
	    #print(correct)

	    print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))  


if __name__== "__main__":
  main()







