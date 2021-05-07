# perceptron-learner
# CSC 426 Project 4
# Authors: 
 Lana Abdelmohsen, Corbin Grosso, Micheal Giordono, Ali Gueye
# Description: 
Implementing a perceptron learning algorithm
# Content Guide and High-level Description of Code
1. project4.py 
- Primarily for executing the code.
- Contains a function for randomly shuffling data in the dataset and records it in a specified output file, it has the parameter output_directory, which is a
path to a directory as an argument called (it is mainly used for task 4). 
- We have a function that reads in the data from the data file given and formats it to be used for training and then a for loop to execute the tasks of the project.
2. model.py 
- contains a model class which contains: 
  - A training function that implements the perceptron training algorithm using the perceptron training rule and also keeps track of the errors and epochs. It's parameters are a list of training examples, an output_directory (path to a file to record all of the training statistics in) and the learning rate as arguments. 
  - A classify function used for classifying the provided training examples based on the current weights (used for getting the target output).
  - A predict function used for predicting the provided training example based on the current weights (used for the output and it calculates the dot product).
  - A function that prints out all the information about the model.
  - A function for graphing the number of errors vs the number of epochs and saves the graph in the specified output directory (has output_directory as a parameter for which is a path to a file to save the graph in).
3. D2
- The "epoch stats file" containing epoch #, # of errors on training data for that epoch, and current weight vector for each of the three LPs for T2. The plot for each of the three LPs for T2.
4. D3
- All the epoch stats files and all the plots from T3.
5. D4
- All the epoch stats files and all the plots from T4.
6. D5
- All the written reports from T5. 
7. The shuffled data files from tasks 4.1 and 4.2
# Instructions for Use
1. Compatibile with python 3.6.0. 
2. TCNJ HPC specific instructions:
- > module add python/3.6.0
3. To run the program:
- > python3 project4.py 
