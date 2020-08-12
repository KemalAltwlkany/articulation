import csv as csv
import os as os
import numpy as np
import pandas as pd
import pickle as pickle
from sklearn.tree import DecisionTreeClassifier, export_text, export_graphviz

def csv_to_arrays(problem_name=None, random_seed=0):
    load_path = '/home/kemal/Programming/Python/Articulation/data/progressive/trees/data/'
    save_path = '/home/kemal/Programming/Python/Articulation/data/progressive/trees/tree_models/'

    # open the csv file, read the columns and convert to <np.ndarray>
    file_name = load_path + problem_name + '_inputs.csv'
    df = pd.read_csv(file_name)
    f1_inputs = df.f1.values.copy()  # f1, f2 and target are <np.ndarray>
    f2_inputs = df.f2.values.copy()
    targets = df.target.values.copy()

    # shuffle the inputs and targets - TOGETHER/SIMULTANEOUSLY
    np.random.seed(random_seed)
    indices = np.arange(len(f1_inputs))
    np.random.shuffle(indices)
    f1_inputs, f2_inputs, targets = f1_inputs[indices], f2_inputs[indices], targets[indices]

    print(f1_inputs)
    print(f2_inputs)
    print(targets)

    #Create matrix of inputs
    inputs = np.array([f1_inputs, f2_inputs]).transpose()
    print(inputs)
    print(targets)
    return [inputs, targets]

def create_decisionTree(problem_name=None, random_seed=0, save=True):
    # load the prepared dataset
    load_path = '/home/kemal/Programming/Python/Articulation/data/progressive/prepared_data/'
    save_path = '/home/kemal/Programming/Python/Articulation/data/progressive/trees/'

    file_name = load_path + 'prepared_dataset_' + problem_name + '.pickle'
    data = None
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    print(data.keys())

    # Shuffle dataset before separating to training and validation dataset
    # Shuffling must be synchronious, i.e. all three arrays are shuffled simultaneously.
    np.random.seed(random_seed)
    indices = np.arange(data['f1_inputs'].size)
    np.random.shuffle(indices)
    data['f1_inputs'] = data['f1_inputs'][indices]
    data['f2_inputs'] = data['f2_inputs'][indices]
    data['targets'] = data['targets'][indices]

    # Create and train tree
    tree = DecisionTreeClassifier()
    inputs = np.stack((data['f1_inputs'], data['f2_inputs']), axis=1)
    targets = data['targets']
    tree.fit(inputs, targets)
    print('Prediction: ')
    print('Expected: I2', ' got: ', tree.predict(np.array([[50, 50]])))
    print('Expected: I12', ' got: ', tree.predict(np.array([[90, 90]])))
    print('Expected: G*', ' got: ', tree.predict(np.array([[10, 10]])))
    print('Expected: G*', ' got: ', tree.predict(np.array([[-1, -1]])))
    print('Expected: G', ' got: ', tree.predict(np.array([[30, 25]])))
    print('Expected: I2', ' got: ', tree.predict(np.array([[10, 55]])))

    #save the model
    if save is True:
        save_name = save_path + 'decision_tree_' + problem_name + '.pickle'
        with open(save_name, 'wb') as f:
            pickle.dump(tree, f, pickle.HIGHEST_PROTOCOL)


def manually_validate_decisionTree(problem_name=None, validation_set='manual'):
    load_path = '/home/kemal/Programming/Python/Articulation/data/progressive/trees/'
    file_name = load_path + 'decision_tree_' + problem_name + '.pickle'
    tree = None
    with open(file_name, 'rb') as f:
        tree = pickle.load(f)

    f1, f2, targets = None, None, None
    # load the manual validation data
    if validation_set is 'manual':
        file_name = '/home/kemal/Programming/Python/Articulation/data/progressive/validation_data/' + problem_name + '_validation_data.csv'
        df = pd.read_csv(file_name)
        f1 = df.f1.values.copy()
        f2 = df.f2.values.copy()
        targets = df.targets.values.copy()


    # predict and compare
    #for i in range(f1.size):
    #    print('Actual: ', targets[i])
    #    print('Predicted: ', tree.predict(np.array([[f1[i], f2[i]]])))
    #    print('-----------------------------------------------------------')
    print(export_text(tree, feature_names=['f1', 'f2']))
    tree.plot()
    #export_graphviz(tree, load_path+'view.png')


if __name__ == '__main__':
    #create_decisionTree('BK1')
    manually_validate_decisionTree('BK1')




