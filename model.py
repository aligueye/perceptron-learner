# Authors: Lana Abdelmohsen, Corbin Grosso, Micheal Giordono, Ali Gueye
# Filename: model.py
# Description: Has the model class which is vital for training our perceptron, and calculating errors and epochs and a function to create a graph.  

import random
import matplotlib.pyplot as plt


class Model:

    def __init__(self, flower, initial_weight_value=0.0, random_weights=False):
        """
		Initialize unlearned model.

		args:
			flower: the model's flower type, either iris setosa, iris versicolor, or iris virginica
            initial_weight_value: Initial value for the weights. Ignored if random_weights=False
            random_weights: Initalizes weights to random value between 1-0. If passed False, initialize all weights to 1
		"""

        self.flower = flower
        self.init_weights(4, initial_weight_value, random_weights)
        self.epochs = 0
        self.errors_after_training = 0
        self.errors_at_each_epoch = []

    def init_weights(self, num_attributes, initial_weight_value=0.0, random_weights=False):
        """
        Initialize weights of unlearned model

        args:
            num_attributes: Number of attributes in each training example
            initial_weight_value: Initial value for the weights. Ignored if random_weights=True
            random_weights: Initalizes weights to random value between 1-0. If passed False, initialize all weights to 1
        """

        if random_weights:
            self.weights = [random.random() for i in range(num_attributes + 1)]
        else:
            self.weights = [0.0 for i in range(num_attributes + 1)]

    def train(self, examples, output_directory, learning_rate=0.01):
        """
        Trains model with provided data. Algorithm implemented is the Perceptron Training Rule

        args:
            examples: List of training examples
            output_directory: Path to a file to record all of the training statistics in
            learning_rate: Rate at which the model learns
        """

        with open(output_directory, 'w') as f:
            f.write(f"Target flower: {self.flower}\n\n")

        while True:
            self.epochs += 1
            self.errors_at_each_epoch.append(0)

            for example in examples:
                target = self.classify(example)
                output = self.predict(example)

                if target is not output: # Weights are only changed if prediction is incorrect
                    
                    self.errors_at_each_epoch[-1] += 1

                    # Update weights
                    for i in range(len(example) - 1):
                        self.weights[i] = self.weights[i] + (learning_rate * (target - output) * example[i])
            
            with open(output_directory, 'a') as f:
                f.write(f"Epoch: {self.epochs}\nAmount of Errors on Current Epoch: {self.errors_at_each_epoch[-1]}\nCurrent Weight Vector: {self.weights}\n\n")
            
            # if no errors are made in the most recent epoch
            if self.errors_at_each_epoch[-1] == 0:
                break
            # if there are more than two epochs completed and the newest three have the same number of errors
            if self.epochs > 2 and self.errors_at_each_epoch[-1] == self.errors_at_each_epoch[-2] and self.errors_at_each_epoch[-1] == self.errors_at_each_epoch[-3]:
                break
            
        self.errors_after_training = self.errors_at_each_epoch[-1]
        with open(output_directory, 'a') as f:
            f.write(self.to_string())


    def classify(self, training_example):
        """
        Classify the provided training example based on the current weights
        args:
            training_example: List of attributes to use to classify the flower type
        """

        if self.flower == training_example[-1]:
            return 1
        else:
            return -1


    def predict(self, training_example):
        """
        Predict the provided training example based on the current weights
        args:
            training_example: List of attributes to use in predict the flower type
        """

        total = 0
        
        for i in range(len(training_example) - 1):
            total += self.weights[i] * training_example[i] # total += w_i * x_i
        if total > 0:
            return 1
        else:
            return -1
    

    def to_string(self):
        """
        Prints out all of the information about the model
        """

        output = f"Target flower: {self.flower}\nWeights: {self.weights}\nEpochs: {self.epochs}\n"
        output += f"Errors in Final Model: {self.errors_after_training}\nErrors Across All Epochs: {sum(self.errors_at_each_epoch)}"
        return output


    def graph_errors_versus_epochs(self, output_directory):
        """
        Creates a graph of the number of errors vs the number of epochs.
        Saves the graph in the specified output directory.

        args:
            output_directory: Path to a file to save the graph as
        """

        if self.epochs == 0:
            raise ValueError("The model has not been trained.\n\
                The model must be trained before a graph can be produced.")
        
        plt.plot([i for i in range(1, self.epochs + 1)], self.errors_at_each_epoch)
        plt.locator_params(axis="both", integer=True, tight=True)
        plt.title('Errors Made Vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Amount of Errors')
        plt.ylim(bottom=0)
        plt.savefig(output_directory)
        plt.clf()

