import numpy as np
from math import exp
from numpy.random import randn
from numpy.random import rand
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# transfer function
def transfer(activation):
	# sigmoid transfer function
	return 1.0 / (1.0 + exp(-activation))

# activation function
def activate(row, weights):
	# add the bias, the last weight
	activation = weights[-1]
	# add the weighted input
	for i in range(len(row)):
		activation += weights[i] * row[i]
	return activation

# activation function for a network
def predict_row(row, network):
	inputs = row
	# enumerate the layers in the network from input to output
	for layer in network:
		new_inputs = list()
		# enumerate nodes in the layer
		for node in layer:
			# activate the node
			activation = activate(inputs, node)
			# transfer activation
			output = transfer(activation)
			# store output
			new_inputs.append(output)
		# output from this layer is input to the next layer
		inputs = new_inputs
	return inputs[0]

# use model weights to generate predictions for a dataset of rows
def predict_dataset(X, network):
	yhats = list()
	for row in X:
		yhat = predict_row(row, network)
		yhats.append(yhat)
	return yhats

# objective function
def objective(X, y, network):
	# generate predictions for dataset
	yhat = predict_dataset(X, network)
	# round the predictions
	yhat = [round(y) for y in yhat]
	# calculate accuracy
	score = accuracy_score(y, yhat)
	return score

# take a step in the search space
def step(network, step_size):
	new_net = list()
	# enumerate layers in the network
	for layer in network:
		new_layer = list()
		# enumerate nodes in this layer
		for node in layer:
			# mutate the node
			new_node = node.copy() + randn(len(node)) * step_size
			# store node in layer
			new_layer.append(new_node)
		# store layer in network
		new_net.append(new_layer)
	return new_net

# hill climbing local search algorithm
def hillclimbing(X, y, objective, solution, n_iter, step_size):
	# evaluate the initial point
	solution_eval = objective(X, y, solution)
	# run the hill climb
	for i in range(n_iter):
		# take a step
		candidate = step(solution, step_size)
		# evaluate candidate point
		candidte_eval = objective(X, y, candidate)
		# check if we should keep the new point
		if candidte_eval >= solution_eval:
			# store the new point
			solution, solution_eval = candidate, candidte_eval
			# report progress
			print('>%d %f' % (i, solution_eval))
	return [solution, solution_eval]

# define dataset
X, y = make_classification(n_samples=80, n_features=5, n_informative=2, n_redundant=1, random_state=5)
# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.033)
# define the total iterations
n_iter = 80
# define the maximum step size
step_size = 0.1
# determine the number of inputs
n_inputs = X.shape[1]
# one hidden layer and an output layer
n_hidden = 10
hidden1 = [rand(n_inputs + 1) for _ in range(n_hidden)]
output1 = [rand(n_hidden + 1)]
network = [hidden1, output1]
# perform the hill climbing search
network, score = hillclimbing(X_train, y_train, objective, network, n_iter, step_size)
print('Done!')
print('Best: %f' % (score))
# generate predictions for the test dataset
yhat = predict_dataset(X_test, network)
# round the predictions
yhat = [round(y) for y in yhat]
# calculate accuracy
score = accuracy_score(y_test, yhat)
print('Test Accuracy: %.5f' % (score * 1000))


print("#########################################################")


 
X = [[.3,6,.8],
     [4,.5,.06],
     [0,.6,8]]

y = [0,1,1]

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(100, 3),
                   random_state=1,learning_rate='constant',max_iter=100,activation='tanh')

clf.fit(X, y)

print('Coef = \n',  clf.coefs_)
print('============================')

print('Prediction  = ',clf.predict([[10,3,10]]))
print('Prediction  = ',clf.predict([[3,7,9]]))

