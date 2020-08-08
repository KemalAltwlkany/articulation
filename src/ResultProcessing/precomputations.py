# Script used to precompute values, i.e. the TNK search space is difficult to visualize neatly without many points
import math as math
import numpy as np
import pickle as pickle
import os as os
import copy as copy

def precompute_TNK_space_and_front(n_samples=100, n_samples_Pareto=1000, save_folder=None, save_name=None):
    if save_folder is None:
        save_folder = '/home/kemal/Programming/Python/Articulation/data/precomputed_data/'
    if save_name is None:
        save_name = 'TNK_data.pickle'

    save_data = dict()

    x1_space = np.linspace(0, math.pi, n_samples)  # should be extra dense, because of the non-convex border
    x2_space = np.linspace(0, math.pi, n_samples)
    f1 = []
    f2 = []
    for x1 in x1_space:
        for x2 in x2_space:
            if math.isclose(x1, 0):
                c1 = math.pow(x2, 2) - 1.1
            else:
                c1 = math.pow(x1, 2) + math.pow(x2, 2) - 1 - 0.1 * math.cos(16 * math.atan(x2 / x1))
            c2 = math.pow(x1 - 0.5, 2) + math.pow(x2 - 0.5, 2) - 0.5
            if c1 >= 0 >= c2:
                f1.append(x1)
                f2.append(x2)
    save_data['search_f1'] = copy.deepcopy(f1)
    save_data['search_f2'] = copy.deepcopy(f2)

    f1 = []
    f2 = []
    x1_space = np.linspace(0, math.pi, n_samples_Pareto)
    x2_space = np.linspace(0, math.pi, n_samples_Pareto)
    for x1 in x1_space:
        for x2 in x2_space:
            if math.isclose(x1, 0):
                c1 = math.pow(x2, 2) - 1.1
            else:
                c1 = math.pow(x1, 2) + math.pow(x2, 2) - 1 - 0.1 * math.cos(16 * math.atan(x2 / x1))
            c2 = math.pow(x1 - 0.5, 2) + math.pow(x2 - 0.5, 2) - 0.5
            if math.isclose(c1, 0, abs_tol=0.001) and 0.0 >= c2:
                f1.append(x1)
                f2.append(x2)

    f1_pom = []
    f2_pom = []
    # this additional nested for-loop is a brute-force method that filters out the actual Pareto optimums.
    # the reason for this is that even though the previous piece of code does find the curve which defines the
    # region border, its non convex and not the entire curve makes up the Pareto front.
    for i in range(len(f1)):
        p1, p2 = f1[i], f2[i]
        trig = False
        for j in range(len(f1)):
            if p1 - f1[j] > 0 and p2 - f2[j] > 0:
                trig = True
                break
            else:
                continue
        if not trig:
            f1_pom.append(p1)
            f2_pom.append(p2)

    save_data['pareto_f1'] = copy.deepcopy(f1_pom)
    save_data['pareto_f2'] = copy.deepcopy(f2_pom)
    os.chdir(save_folder)
    with open(save_name, 'wb') as f:
        pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    precompute_TNK_space_and_front(n_samples=200, n_samples_Pareto=1100)

