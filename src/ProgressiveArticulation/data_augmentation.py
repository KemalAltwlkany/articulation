import csv as csv
import os as os
import pickle as pickle
import numpy as np


def augment_data(benchmark_problem='BK1'):
    load_path = '/home/kemal/Programming/Python/Articulation/data/progressive/articulatedData/'
    save_path = '/home/kemal/Programming/Python/Articulation/data/progressive/augmentedData/'


    file_name = load_path + benchmark_problem + '_inputs.csv'
    lst = []
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            print(row['val_f1'], row['dec_f1'], row['val_f2'], row['dec_f2'])
            lst.append([row['val_f1'], row['dec_f1'], row['val_f2'], row['dec_f2']])
    print(lst)



if __name__ == '__main__':
    augment_data()



