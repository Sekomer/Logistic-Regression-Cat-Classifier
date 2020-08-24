# CAT IMAGE CLASSIFICATION
# I'll add sys.argv input option for external image classification

#Cost after iteration 1900: 0.407786610208405
#train accuracy: 89.47368421052632 %
#test accuracy: 84.0 %

import numpy as np
import matplotlib.pyplot as plt
import load_dataset_py 

# Answer to the Ultimate Question of Life, the Universe, and Everything LOL
np.random.seed(42)

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset_py.load_dataset()

# Reshape the training and test examples
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.


def sigmoid(z):
	s = 1 / ( 1 + np.exp(-z))
	return s

def initialize(dim):
	w = np.ones((dim,1))*0.1
	b = 0/1
	
	return w, b

def propagate(w, b, X, Y):
	m = X.shape[1]
	
	A = sigmoid(np.dot(w.T, X)+b)          # compute activation
	cost = (-1/m)*np.sum(Y*np.log(A) + (1-Y)*np.log(1-A)) + np.sum(np.abs(w))*0.001    # compute cost, i tried to implement regularization
	
	# BACKWARD PROPAGATION
	dw = np.dot(X,(A-Y).T)/m
	db = np.sum(A-Y)/m
	cost = np.squeeze(cost) #Removing single-dimensional entries 

	grads = {"dw": dw,
			 "db": db}
	
	return grads, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
	costs = []
	
	for i in range(num_iterations):
		# Cost and gradient calculation
		grads, cost = propagate(w, b, X, Y)

		# Retrieve derivatives from grads
		dw = grads["dw"]
		db = grads["db"]

		w = w - learning_rate*dw
		b = b - learning_rate*db

		# Recording and printing the cost every 100 training iterations
		if i % 100 == 0:
			costs.append(cost)

		if print_cost and i % 100 == 0:
			print ("Cost after iteration {}: {}".format(i, cost))
	
	params = {"w": w,
			  "b": b}
	
	grads = {"dw": dw,
			 "db": db}
	
	return params, grads, costs

# makes prediction with current wweight and bias
def predict(w, b, X):
	m = X.shape[1]
	Y_prediction = np.zeros((1,m))
	w = w.reshape(X.shape[0], 1)
	
	A = sigmoid(np.dot(w.T, X)+b) + 0.1
	
	for i in range(A.shape[1]):
		if A[0,i] >= 0.50:
			Y_prediction[0,i] = 1 
		else:
			Y_prediction[0,i] = 0
	
	return Y_prediction

# this is optional, to use set make learn_curve in model's arguments to True
# Plots learning curve with costs
def plot_learn_curve(dd):	
	costs = np.squeeze(dd['costs'])
	plt.plot(costs)
	plt.ylabel('cost')
	plt.xlabel('iterations (per hundreds)')
	plt.title("Learning rate =" + str(dd["learning_rate"]))
	plt.show()

# creating model
def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.001, print_cost = False, learn_curve = False):
	
	w, b = initialize(X_train.shape[0])

	parameters, _ , costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
	
	w = parameters["w"]
	b = parameters["b"]
	
	Y_prediction_test = predict(w, b, X_test)
	Y_prediction_train = predict(w, b, X_train)

	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))


	d = {"costs": costs,
		 "Y_prediction_test": Y_prediction_test, 
		 "Y_prediction_train" : Y_prediction_train, 
		 "w" : w, 
		 "b" : b,
		 "learning_rate" : learning_rate,
		 "num_iterations": num_iterations}
	
	if learn_curve:
		plot_learn_curve(d)

	return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000 , 
						learning_rate = 0.001, print_cost = True, learn_curve = True)

