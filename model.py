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
        self.init_weights()

    def init_weights(self, random_weights = True):
        """
        Initialize weights of unlearned model

        args:
            random: Initalizes weights to random value between 1-0. If passed False, initialize all weights to 1
        """

        if random_weights:
            self.weights = [random.random() for i in range(4)]
        else:
            self.weights = [1.0 for i in range(4)]

    def get_training_data(self, shuffle = False):
        """
        Get training data into model

        args:
            shuffle: Determines if data should be shuffles. If passed True, shuffle training data randomly
        """

        pass
    
    def to_string(self):
        print(f"Target flower: {self.flower}")
        print(f"Learned weights: {self.weights}")

def main():
    model = Model("Iris-setosa")
    model.to_string()

if __name__ == "__main__":
	main()
