# Authors: Lana Abdelmohsen, Corbin Grosso, Micheal Giordono, Ali Gueye
# Filename: model.py
# Description: Perceptron model

import random

class Model:

    def __init__(self, flower):
        """
		Initialize unlearned model.

		args:
			flower : the model's flower type, either iris setosa, iris versicolor, or iris virginica
		"""
        self.flower = flower
        self.init_weights(4)
        self.errors = 0
        self.epochs = 0

        # self.train(self)

    def init_weights(self, num_attributes, random_weights = False):
        """
        Initialize weights of unlearned model

        args:
            num_attributes: Number of attributes in each training example
            random: Initalizes weights to random value between 1-0. If passed False, initialize all weights to 1
        """
        if random_weights:
            self.weights = [random.random() for i in range(num_attributes + 1)]
        else:
            self.weights = [0.0 for i in range(num_attributes + 1)]

    def train(self, examples, shuffle = False):
        """
        Trains model with provided data. Algorithm implemented is Stochastic Gradient Descent (SGD)
        args:
            shuffle: Determines if data should be shuffles. If passed True, shuffle training data randomly
        """

        # Randomly shuffles examples
        if shuffle:
            random.shuffle(examples)

        errors_in_epoch = 1 # set to 1 to assure training loop executes
        errors_in_prev = 0

        while (errors_in_epoch > 0) ^ (errors_in_prev == errors_in_epoch):
            if(self.epochs != 0):
                errors_in_prev = errors_in_epoch

            self.epochs += 1
            errors_in_epoch = 0

            for example in examples:
                target = self.classify(example)
                output = self.predict(example)

                if target is not output: # Weights are only changed if prediction is incorrect
                    
                    self.errors += 1
                    errors_in_epoch +=1

                    for i in range(len(example) - 1):
                        self.weights[i] = self.weights[i] + (0.01 * (target - output) * example[i])
            print(f"Weights: {self.weights}")
            print(f"Epochs: {self.epochs}")
            print(f"Errors {errors_in_epoch}")
            print(f"errors_in_prev = errors_in_epoch: {errors_in_prev == errors_in_epoch}")


    def classify(self, training_example):
        """
        Classify the provided training example based on the current weights
        args:
            training_example: List of attributes to use to classify the flower type
        """
        # print(f"flower: {self.flower}")
        # print(f"curr flower: {training_example[-1]}")
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
        print(f"Target flower: {self.flower}")
        print(f"Weights: {self.weights}")
        print(f"Epochs: {self.epochs}")
        print(f"Errors {self.errors}")

# def main():
#     model = Model("Iris-setosa")
#     model.to_string()

# if __name__ == "__main__":	
#     main()
