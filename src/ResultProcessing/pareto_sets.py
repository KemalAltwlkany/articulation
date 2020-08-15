import numpy as np
import os as os
import pickle as pickle



# this script will precompute and save a np.array of the Pareto sets for every problem.

def BK1_pareto_set(n_points_per_axis=100):
    save_path = '/home/kemal/Programming/Python/Articulation/data/performance_measures/pareto_sets/'
    # The objective space - a few samples
    x1_range = np.linspace(-5, 10, n_points_per_axis)
    x2_range = np.linspace(-5, 10, n_points_per_axis)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    f1 = x1 ** 2 + x2 ** 2
    f2 = (x1 - 5) ** 2 + (x2 - 5) ** 2
    f1 = np.reshape(f1, f1.size)
    f2 = np.reshape(f2, f2.size)
    x1 = np.reshape(x1, x1.size)
    x2 = np.reshape(x2, x2.size)




if __name__ == '__main__':
    BK1_pareto_set(n_points_per_axis=5)


