import numpy as np
import os as os
import pandas as pd
import pickle as pickle


# RBDT - no implementation was found on the internet.
# Might implement it myself (the RBDT)
# Entire class is hardcoded due to lack of time for implementing the RBDT algorithm.
class RuleBasedTree:
    def __init__(self, f1=None, f2=None):
        # d is a list of dictionaries.
        self.f1_A = f1[0]
        self.f1_B = f1[1]
        self.f1_C = f1[2]
        self.f1_D = f1[3]
        self.f1_E = f1[4]
        self.f1_F = f1[5]

        self.f2_A = f2[0]
        self.f2_B = f2[1]
        self.f2_C = f2[2]
        self.f2_D = f2[3]
        self.f2_E = f2[4]
        self.f2_F = f2[5]

    def classify(self, y):
        # Returns class of solution based on the values of its criteria
        f1, f2 = y[0], y[1]
        if f1 > self.f1_C and f2 > self.f2_C:
            return 'I12'
        elif f1 < self.f1_C and f2 < self.f2_C:
            return 'G'
        elif f1 > self.f1_D:
            return 'I1*'
        elif f2 > self.f2_D:
            return 'I2*'
        elif f1 > self.f2_C:
            return 'I1'
        else:
            return 'I2'


def train_RBTree(problem_name=None, save=True, ret=False, which_csv_index='0'):
    load_path = '/home/kemal/Programming/Python/Articulation/data/progressive/articulated_data/'
    save_path = '/home/kemal/Programming/Python/Articulation/data/progressive/RBTrees/'

    # open the csv file, read the columns and convert to dictionaries for easier coding
    file_name = load_path + '/' + problem_name + '/' + problem_name + '_' + which_csv_index + '_articulated.csv'
    df = pd.read_csv(file_name)
    f1 = df.f1.values.copy()
    f2 = df.f2.values.copy()
    limits = df.limits.values.copy()
    tree = RuleBasedTree(f1, f2)
    if save is True:
        save_name = save_path + problem_name + '_rbtree.pickle'
        with open(save_name, 'wb') as f:
            pickle.dump(tree, f, pickle.HIGHEST_PROTOCOL)

    if ret is True:
        # returns f1 and f2 as dictionaries with keys defined by limits
        f1 = dict(zip(limits, f1))
        f2 = dict(zip(limits, f2))
        return tree, f1, f2


if __name__ == '__main__':
    train_RBTree('BK1', save=False, ret=False)


