# Authors: Lana Abdelmohsen, Corbin Grosso, Micheal Giordono, Ali Gueye
# Filename: project4.py
#Description: Main file for execution of all required tasks
from model import Model
import random


def shuffle_data(output_directory):
    """
    Shuffles the data in the dataset and records it in the specified output file

    args:
        output_directory: path to a file to record all of the shuffled data
    """

    random.shuffle(dataset)
    with open(output_directory, 'w') as f:
        for example in dataset:
            line = ''
            for attr in example[1:]:
                line += f"{attr},"
            line = line[:-1] # remove trailing comma
            line += '\n'
            f.write(line)


def read_in_data(input_directory):
    """
    Reads in data from the specified input file and formats it to be used for training

    args:
        input_directory: path to a file containing data to use for training
    """

    # reads data
    with open(input_directory,'r') as f:
        lines = f.readlines()
    for line in lines:
        dataset.append(line.split(','))

    # format data
    for row in dataset:
        row[-1] = row[-1].strip()
        for i in range(4):
            row[i] = float(row[i])
        row.insert(0, 1) # Makes the first value 1, pushes every other attribute over by 1, represents w_0


dataset = []

read_in_data('iris.data')

# Naming scheme used for the model objects is
# model_{name of target flower}_{task the model is for, with decimals replaced with underscores}

# Task 2: Trains model for three different learning problems
print("Task 2:\n")
model_setosa_2 = Model('Iris-setosa')
model_setosa_2.train(dataset)
model_setosa_2.to_string()
model_setosa_2.graph_errors_versus_epochs('graphs/task_2_setosa.pdf')

print()

model_versicolor_2 = Model('Iris-versicolor')
model_versicolor_2.train(dataset)
model_versicolor_2.to_string()
model_versicolor_2.graph_errors_versus_epochs('graphs/task_2_versicolor.pdf')

print()

model_virginica_2 = Model('Iris-virginica')
model_virginica_2.train(dataset)
model_virginica_2.to_string()
model_virginica_2.graph_errors_versus_epochs('graphs/task_2_virginica.pdf')

print()


# Task 3.1: Trains models with initial weights all set to 1
print("Task 3.1:\n")
model_setosa_3_1 = Model('Iris-setosa', 1)
model_setosa_3_1.train(dataset)
model_setosa_3_1.to_string()
model_setosa_3_1.graph_errors_versus_epochs('graphs/task_3.1_setosa.pdf')

print()

model_versicolor_3_1 = Model('Iris-versicolor', 1)
model_versicolor_3_1.train(dataset)
model_versicolor_3_1.to_string()
model_versicolor_3_1.graph_errors_versus_epochs('graphs/task_3.1_versicolor.pdf')

print()

model_virginica_3_1 = Model('Iris-virginica', 1)
model_virginica_3_1.train(dataset)
model_virginica_3_1.to_string()
model_virginica_3_1.graph_errors_versus_epochs('graphs/task_3.1_virginica.pdf')

print()


# Task 3.2: Trains models with four random weights between 0 and 1
print("Task 3.2:\n")
model_setosa_3_2 = Model('Iris-setosa', random_weights=True)
model_setosa_3_2.train(dataset)
model_setosa_3_2.to_string()
model_setosa_3_2.graph_errors_versus_epochs('graphs/task_3.2_setosa.pdf')

print()

model_versicolor_3_2 = Model('Iris-versicolor', random_weights=True)
model_versicolor_3_2.train(dataset)
model_versicolor_3_2.to_string()
model_versicolor_3_2.graph_errors_versus_epochs('graphs/task_3.2_versicolor.pdf')

print()

model_virginica_3_2 = Model('Iris-virginica', random_weights=True)
model_virginica_3_2.train(dataset)
model_virginica_3_2.to_string()
model_virginica_3_2.graph_errors_versus_epochs('graphs/task_3.2_virginica.pdf')

print()


# Task 3.3: Trains models with four random weights between 0 and 1
print("Task 3.3:\n")
model_setosa_3_3 = Model('Iris-setosa', random_weights=True)
model_setosa_3_3.train(dataset)
model_setosa_3_3.to_string()
model_setosa_3_3.graph_errors_versus_epochs('graphs/task_3.3_setosa.pdf')

print()

model_versicolor_3_3 = Model('Iris-versicolor', random_weights=True)
model_versicolor_3_3.train(dataset)
model_versicolor_3_3.to_string()
model_versicolor_3_3.graph_errors_versus_epochs('graphs/task_3.3_versicolor.pdf')

print()

model_virginica_3_3 = Model('Iris-virginica', random_weights=True)
model_virginica_3_3.train(dataset)
model_virginica_3_3.to_string()
model_virginica_3_3.graph_errors_versus_epochs('graphs/task_3.3_virginica.pdf')

print()


# Task 4.1: Trains model for three different learning problems
shuffle_data('iris_shuffle_for_T4.1.data')

print("Task 4.1:\n")
model_setosa_4_1 = Model('Iris-setosa')
model_setosa_4_1.train(dataset)
model_setosa_4_1.to_string()
model_setosa_4_1.graph_errors_versus_epochs('graphs/task_4.1_setosa.pdf')

print()

model_versicolor_4_1 = Model('Iris-versicolor')
model_versicolor_4_1.train(dataset)
model_versicolor_4_1.to_string()
model_versicolor_4_1.graph_errors_versus_epochs('graphs/task_4.1_versicolor.pdf')

print()

model_virginica_4_1 = Model('Iris-virginica')
model_virginica_4_1.train(dataset)
model_virginica_4_1.to_string()
model_virginica_4_1.graph_errors_versus_epochs('graphs/task_4.1_virginica.pdf')

print()


# Task 4.2: Trains model for three different learning problems
shuffle_data('iris_shuffle_for_T4.2.data')

print("Task 4.2:\n")
model_setosa_4_2 = Model('Iris-setosa')
model_setosa_4_2.train(dataset)
model_setosa_4_2.to_string()
model_setosa_4_2.graph_errors_versus_epochs('graphs/task_4.2_setosa.pdf')

print()

model_versicolor_4_2 = Model('Iris-versicolor')
model_versicolor_4_2.train(dataset)
model_versicolor_4_2.to_string()
model_versicolor_4_2.graph_errors_versus_epochs('graphs/task_4.2_versicolor.pdf')

print()

model_virginica_4_2 = Model('Iris-virginica')
model_virginica_4_2.train(dataset)
model_virginica_4_2.to_string()
model_virginica_4_2.graph_errors_versus_epochs('graphs/task_4.2_virginica.pdf')

print()
