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

    def train(self, vectors, values):
      """
      vectors : Training vectors, vectors.shape in the form of [samples, #features]
      values: Target values, values.shape in the form of [#samples]
      """

      for i in range(self.niterations): #for all misclassifications 
         error= 0 #error counter
         for xi, target in zip(vectors, values): #for x in the training vectors and values 
            delta_w = self.rate * (target - self.predict(xi)) #calculate the approperiate calculations for w and update count
            self.weight[1:] += delta_w * xi
            self.weight[0] += delta_w
            count += int(delta_w != 0.0)
         self.errors.append(error) #append error counter 
      return self
	
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
