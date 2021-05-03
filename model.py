# Authors: Lana Abdelmohsen, Corbin Grosso, Micheal Giordono, Ali Gueye
# Filename: model.py
# Description: Perceptron model

import random

class Model:

    def __init__(self, flower,rate = 0.01, niterations = 10):
        """
		Initialize unlearned model.

		args:
			flower : the model's flower type, either iris setosa, iris versicolor, or iris virginica
		"""

        self.flower = flower
        self.init_weights() 
	self.rate = rate
        self.Iterations = niterations
        self.errors = []

    def init_weights(self, num_attributes, random_weights = True):
        """
        Initialize weights of unlearned model

        args:
            num_attributes: Number of attributes in each training example
            random: Initalizes weights to random value between 1-0. If passed False, initialize all weights to 1
        """

        if random_weights:
            self.weights = [random.random() for i in range(num_attributes + 1)]
        else:
            self.weights = [1.0 for i in range(num_attributes + 1)]

    def get_training_data(self, shuffle = False):
        """
        Get training data into model

        args:
            shuffle: Determines if data should be shuffles. If passed True, shuffle training data randomly
        """

        pass

    def train_weights(train, l_rate, n_epoch):
      """
      l_rate: takes the learning rate as a parameter
      n_epoch: takes the epoch as a parameter
      """ 
# Estimate Perceptron weights using stochastic gradient descent
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		for row in train:
			prediction = classify(row, weights)
			error = row[-1] - prediction
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
	return weights
 
 	
    def perceptron(train, test, l_rate, n_epoch)
      """
      l_rate: takes the learning rate as a parameter
      n_epoch: takes the epoch as a parameter
      """ 
#Preceptron Algorithm weights using stochastic gradient descent
	predictions = list()
	weights = train_weights(train, l_rate, n_epoch)
	for row in test:
		prediction = classify(row, weights)
		predictions.append(prediction)
	return(predictions)
      
	
     def classify(self, training_example):
        """
        Classify the provided training example based on the current weights

        args:
            training_example: List of attributes to use in classify the flower type
        """

        total = 0
        training_example.insert(0, 1) # Makes the first value 1, pushes every other attribute over by 1
        for i in range(len(training_example)):
            total += self.weights[i] * training_example[i] # total += w_i * x_i
        if total > 0:
            return 1
        else:
            return -1
    
    def to_string(self):
        print(f"Target flower: {self.flower}")
        print(f"Learned weights: {self.weights}")

def main():
    model = Model("Iris-setosa")
    model.to_string()

if __name__ == "__main__":
	main()
