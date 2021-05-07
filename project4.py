# Authors: Lana Abdelmohsen, Corbin Grosso, Micheal Giordono, Ali Gueye
# Filename: project4.py
#Description: Main file for execution of all required tasks (includes a shuffle method for task 4 to shuffle the data in the iris-data file)
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
#for loop for executing details of different tasks like shuffling for 4.1 and 4.2
for task in ['2', '3.1', '3.2', '3.3', '4.1', '4.2']: 
    if task == '4.1' or task == '4.2':  
        shuffle_data(f'iris_shuffle_for_T{task}.data') 
    for species in ['setosa', 'versicolor', 'virginica']: 
        if task == '3.1':  
            model = Model(f'Iris-{species}', 1)  
        elif task == '3.2' or task == '3.3': 
            model = Model(f'Iris-{species}', random_weights=True) 
        else:
            model = Model(f'Iris-{species}')
        
        model.train(dataset, f'run_stats/task_{task}_{species}.txt') 
        model.graph_errors_versus_epochs(f'graphs/task_{task}_{species}.pdf')
