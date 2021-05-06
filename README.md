# perceptron-learner
# CSC 426 Project 4
# Authors: 
Lana Abdelmohsen,
# Description: 
A basic descision tree learning algorithm, capable of learning a tree from multiple data sources and producing a visual output.
# Content Guide and High-level Description of Code
1. project4.py 
- primarily for executing the code
- 
2. model.py 
- 
- contains the Node class, which
   - is a simple, non-optimized n-tree with labeled edges
   - serves as the backbone for the DecisionTree class
   - can convert itself into a string representation
3. D2
- The "epoch stats file" containing epoch #, # of errors on training data for that epoch, and current weight vector for each of the three LPs for T2. The plot for each of the three LPs for T2.
4. D3
- All the epoch stats files and all the plots from T3.
5. D4
- All the epoch stats files and all the plots from T4.
6. D5
- All the written reports from T5.
# Instructions for Use
1. Compatibile with python 3.6.0. 
2. TCNJ HPC specific instructions:
- > module add python/3.6.0
3.Libraries that might be needed
- > pip install matplotlib
4.  To run the program:
- > python3 project4.py 
