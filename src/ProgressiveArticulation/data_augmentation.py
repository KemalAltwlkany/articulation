import pandas as pd
import os as os
import pickle as pickle
import numpy as np
import random as random


def augment_data(problem_name=None, random_seed=0, n_samples=100, save=True):

    # Function used to create a region f1 € [p1, p2], f2 € [q1, q2]
    # number of samples is n, and the zone is labelled by sig
    def create_region(sig=None, p1=None, p2=None, q1=None, q2=None, n=None):
        f1_region = np.linspace(f1[p1], f1[p2], n)
        f2_region = np.linspace(f2[q1], f2[q2], n)
        xx, yy = np.meshgrid(f1_region, f2_region)
        xx = np.reshape(xx, xx.size)
        yy = np.reshape(yy, yy.size)
        labels = np.array([sig] * xx.size)
        return xx, yy, labels

    load_path = '/home/kemal/Programming/Python/Articulation/data/progressive/articulated_data/'
    save_path = '/home/kemal/Programming/Python/Articulation/data/progressive/prepared_data/'

    # open the csv file, read the columns and convert to dictionaries for easier coding
    file_name = load_path + problem_name + '_articulated.csv'
    df = pd.read_csv(file_name)
    f1 = df.f1.values.copy()
    f2 = df.f2.values.copy()
    limits = list(df.limits.values.copy())
    f1 = dict(zip(limits, f1))
    f2 = dict(zip(limits, f2))
    print(f1)
    print(f2)

    # Part where the data gets augmented
    np.random.seed(random_seed)

    # list containing all information regarding the zones:
    lst = [
        ['G*', 'A', 'B', 'A', 'B', n_samples*6],  # 1
        ['G', 'B', 'C', 'A', 'C', n_samples*6],  # 2
        ['G', 'A', 'B', 'B', 'C', n_samples*6],  # 3
        #['G', 'C', 'D', 'C', 'D', n_samples],  # 4
        ['I1', 'C', 'D', 'A', 'C', n_samples*6],  # 5
        ['I2', 'A', 'C', 'C', 'D', n_samples*6],  # 6
        ['I1*', 'D', 'F', 'A', 'C', n_samples*6],  # 7
        ['I2*', 'A', 'C', 'D', 'F', n_samples*6],  # 8
        ['I12', 'C', 'F', 'C', 'F', n_samples*10]  # 9
    ]


    f1_inputs = np.array([])
    f2_inputs = np.array([])
    targets = np.array([])
    for i in lst:
        x, y, z = create_region(*i)
        f1_inputs = np.concatenate((f1_inputs, x))
        f2_inputs = np.concatenate((f2_inputs, y))
        targets = np.concatenate((targets, z))

    print(f1_inputs.size)

    # Save data into dictionary and save for decision tree to use.
    if save is True:
        save_data = dict(
            f1_inputs=f1_inputs,
            f2_inputs=f2_inputs,
            targets=targets
        )
        save_name = 'prepared_dataset_' + problem_name + '.pickle'
        os.chdir(save_path)
        with open(save_name, 'wb') as f:
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    augment_data(problem_name='BK1', random_seed=0, n_samples=5, save=True)



