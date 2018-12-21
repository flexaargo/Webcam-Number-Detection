# Based on the neural network created in the book
# Make Your Own Neural Network by Tariq Rashid

import numpy as np
import scipy.special

class NeuralNetwork():
	def __init__(self, input_nodes, hidden_nodes, ouput_nodes, learning_rate):
		# Initialize vars
		self.input_nodes = input_nodes
		self.hidden_nodes = hidden_nodes
		self.output_nodes = ouput_nodes
		self.learning_rate = learning_rate

		# Create the weights matrix between input and hidden nodes
		self.wih = np.random.normal(0.0, pow(self.input_nodes, -0.5), (self.hidden_nodes, self.input_nodes))
		# Create the weights matrix between the hidden and output nodes
		self.who = np.random.normal(0.0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes))

		# Activate function is the sigmoid function
		self.activation_function = lambda x: scipy.special.expit(x)
		pass

	def train(self, inputs_list, targets_list):
		# Convert the inputs list into 2d array
		inputs = np.array(inputs_list, ndmin=2).T
		# Convert the targets list into 2d array
		targets = np.array(targets_list, ndmin=2).T

		# Calculate the signals into the hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# Calculate the signals emerging from the hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		# Calculate the signals into the final output layer
		final_inputs = np.dot(self.who, hidden_outputs)
		# Calculate the signals emerging from the final output layer
		final_outputs = self.activation_function(final_inputs)

		# Calculate the error
		output_errors = targets - final_outputs
		# Calculate hidden layer error
		hidden_errors = np.dot(self.who.T, output_errors)

		# Update the weights between the hidden and output layers
		self.who += self.learning_rate * np.dot((output_errors * final_outputs * (1 - final_outputs)), hidden_outputs.T)
		# Update the weights between the input and hidden layers
		self.wih += self.learning_rate * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), inputs.T)

	def query(self, inputs_list):
		# Convert inputs list into 2d array
		inputs = np.array(inputs_list, ndmin=2).T
		# Calculate the signals into hidden layer
		hidden_inputs = np.dot(self.wih, inputs)
		# Calculate the signals emerging from the hidden layer
		hidden_ouputs = self.activation_function(hidden_inputs)

		# Calculate the signals into final output layer
		final_inputs = np.dot(self.who, hidden_ouputs)
		# Calculate the signals emerging from the final output layer
		final_outputs = self.activation_function(final_inputs)

		return final_outputs
