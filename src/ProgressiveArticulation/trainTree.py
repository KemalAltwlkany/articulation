import csv as csv
import os as os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

def csv_to_arrays(problem_name=None, random_seed=0):
    load_path = '/home/kemal/Programming/Python/Articulation/data/progressive/trees/data/'
    save_path = '/home/kemal/Programming/Python/Articulation/data/progressive/trees/tree_models/'

    # open the csv file, read the columns and convert to <np.ndarray>
    file_name = load_path + problem_name + '_inputs.csv'
    df = pd.read_csv(file_name)
    f1_inputs = df.f1.values.copy()  # f1, f2 and target are <np.ndarray>
    f2_inputs = df.f2.values.copy()
    f1_targets = df.target.values.copy()
    f2_targets = df.target.values.copy()


    # shuffle the inputs and targets - TOGETHER/SIMULTANEOUSLY
    np.random.seed(random_seed)
    indices = np.arange(len(f1_inputs))
    np.random.shuffle(indices)
    f1_inputs, f1_targets = f1_inputs[indices], f1_targets[indices]

    # shuffle again
    np.random.shuffle(indices)
    f2_inputs, f2_targets = f2_inputs[indices], f2_targets[indices]

    print(f1_inputs)
    print(f1_targets)
    print(f2_inputs)
    print(f2_targets)

    #Create matrices of inputs and targets
    inputs = np.array([f1_inputs, f2_inputs]).transpose()
    targets = np.array([f1_targets, f2_targets]).transpose()
    print(inputs)
    print(targets)
    return [inputs, targets]

def create_decisionTree(problem_name=None, data=None):
    tree = DecisionTreeClassifier()
    tree.fit(data[0], data[1])
    print('Prediction: ')
    print(tree.predict(np.array([[50, 50]])))
    print(tree.predict())
    print(tree.predict(np.array([[30, 40]])))





if __name__ == '__main__':
    create_decisionTree('BK1', csv_to_arrays(problem_name='BK1'))





