# Authors: Lana Abdelmohsen, Corbin Grosso, Micheal Giordono, Ali Gueye
# Filename: project4.py
#Description: Main file for execution of all required tasks
from model import Model

# reads data
dataset = []
file = open('iris.data','r')
for line in file:
    dataset.append(line.split(','))
file.close()

# format data
for row in dataset:
    row[-1] = row[-1].strip()
    for i in range(4):
        row[i] = float(row[i])
    row.insert(0, 1) # Makes the first value 1, pushes every other attribute over by 1, represents w_0

# Task 1: Trains model for three different learning problems
model_1_1 = Model('Iris-setosa')
model_1_1.train(dataset)
model_1_1.to_string()

print()

model_1_1 = Model('Iris-versicolor')
model_1_1.train(dataset)
model_1_1.to_string()

print()

model_1_1 = Model('Iris-virginica')
model_1_1.train(dataset)
model_1_1.to_string()
