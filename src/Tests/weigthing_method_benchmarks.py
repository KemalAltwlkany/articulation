import numpy as np
import sys as sys
import os as os

sys.path.insert(0, "/home/kemal/Programming/Python/Articulation")
from src.PreferenceArticulation.Solution import Solution
from src.PreferenceArticulation.BenchmarkObjectives import *
from src.TabuSearch.weighting_method import AposterioriWeightingMethod


def BK1_test(params):
    SearchInstance = AposterioriWeightingMethod(**params)
    SearchInstance.search()


def main():
    params = dict(
        init_sol=Solution(np.array([9, 9])),
        problem=MOO_Problem.BK1,
        constraints=[MOO_Constraints.BK1_constraint],
        step_size=0.05,
        neighborhood_size=15,
        max_iter=2000,
        M=100,
        tabu_list_max_length=20,
        max_loops=100,
        search_space_dimensions=2,
        objective_space_dimensions=2,
        save=True,
        weights=[0.5, 0.5]
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/BK1/',
        filename=''
    )
    for i in range(20):
        save_options['filename'] = 'BK1_test_' + str(i+1) + '.pickle'
        params['save_options'] = save_options
        params['seed_value'] = i
        params['test_ID'] = 'BK1_test_' + str(i+1)
        BK1_test(params)


if __name__ == '__main__':
    main()


