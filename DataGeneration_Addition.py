import numpy as np 


def reverse(x):

	numbers = [0,0,0]
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


	number_list = reverse(x)

	v3[number_list[0]] = 1
	v2[number_list[1]] = 1
	v1[number_list[2]] = 1

	return v1 + v2 + v3

def gen_data(c, d, length):

	data = []
	while (length):
		if c > 999:
			break
		data.append(c)
		c = c + d
		length-=1
	return data 
