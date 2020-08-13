import numpy as np
n_tests = 20
fifth = n_tests//5
which_asp = np.concatenate((np.array([0] * fifth), np.array([1] * fifth), np.array([2] * fifth), np.array([3] * fifth), np.array([4] * fifth)))
print(which_asp)