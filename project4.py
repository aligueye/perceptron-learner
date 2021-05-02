
import numpy as np

class Perceptron:
   def __init__(self, rate = 0.01, niter = 10):
      self.rate = rate
      self.niterations = niterations

   def train(self, vectors, values):
      """
      vectors : Training vectors, vectors.shape in the form of [samples, #features]
      values: Target values, values.shape in the form of [#samples]
      """

      # assign weights
      self.weight = np.zeros(1 + vectors.shape[1]) #np.zeros() returns a new array of given shape and type, where the element's value as 0.

      # Number of misclassifications
      self.errors = []  # Number of misclassifications

      for i in range(self.niterations): #for all misclassifications 
         error= 0 #error counter
         for x, target in zip(vectors, values): #for x in the training vectors and values 
            delta_w = self.rate * (target - self.predict(xi)) #calculate the approperiate calculations for w and update count
            self.weight[1:] += delta_w * xi
            self.weight[0] += delta_w
            count += int(delta_w != 0.0)
         self.errors.append(error) #append error counter 
      return self

   def dot_product(self, vectors):
      """Calculate the dot profuc """
      return np.dot(vectors, self.weight[1:]) + self.weight[0]

   def predict(self, vectors):
      """Return class label after unit step"""
      return np.where(self.net_input(vectors) >= 0.0, 1, -1)
