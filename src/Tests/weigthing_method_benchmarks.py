import numpy as np
import sys as sys
import random as random

sys.path.insert(0, "/home/kemal/Programming/Python/Articulation")
from src.PreferenceArticulation.Solution import Solution
from src.PreferenceArticulation.BenchmarkObjectives import *
from src.TabuSearch.weighting_method import AposterioriWeightingMethod


def aposteriori_FON(n_tests=20, n_dims=2):
    params = dict(
        init_sol=Solution(np.array([3.5]*n_dims)),
        problem=MOO_Problem.FON,
        constraints=[],  # no constraints for this problem
        step_size=0.05,
        neighborhood_size=15,
        max_iter=2000,
        M=100,
        tabu_list_max_length=20,
        max_loops=100,
        search_space_dimensions=n_dims,
        objective_space_dimensions=2,
        save=True,
        weights=[1./n_dims] * n_dims
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/FON/',
        filename=''
    )
    for i in range(n_tests):
        save_options['filename'] = 'FON_test_' + str(i+1) + '.pickle'
        params['save_options'] = save_options
        params['seed_value'] = i
        params['test_ID'] = 'FON_test_' + str(i+1)
        random.seed(i)
        a = random.random()
        params['weights'] = [a, 1 - a]
        np.random.seed(i)
        params['init_sol'] = Solution(np.random.uniform(low=-4, high=4, size=n_dims))
        SearchInstance = AposterioriWeightingMethod(**params)
        SearchInstance.search()


def aposteriori_SCH1(n_tests=20):
    params = dict(
        init_sol=Solution(np.array([10])),
        problem=MOO_Problem.SCH1,
        constraints=[],  # no constraints for this problem
        step_size=0.02,
        neighborhood_size=15,
        max_iter=2000,
        M=100,
        tabu_list_max_length=20,
        max_loops=100,
        search_space_dimensions=1,
        objective_space_dimensions=2,
        save=True,
        weights=[0.5, 0.5]
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/SCH1/',
        filename=''
    )
    for i in range(n_tests):
        save_options['filename'] = 'SCH1_test_' + str(i+1) + '.pickle'
        params['save_options'] = save_options
        params['seed_value'] = i
        params['test_ID'] = 'SCH1_test_' + str(i+1)
        random.seed(i)
        a = random.random()
        params['weights'] = [a, 1-a]
        # Since the search space is 1D, pick initial point as either -10 or 10.
        if i % 2 is 0:
            params['init_sol'] = Solution(np.array([-5 + random.uniform(-1, 1)]))
        else:
            params['init_sol'] = Solution(np.array([5 + random.uniform(-1, 1)]))
        SearchInstance = AposterioriWeightingMethod(**params)
        SearchInstance.search()


def aposteriori_IM1(n_tests=20):
    params = dict(
        init_sol=Solution(np.array([2, 1.5])),
        problem=MOO_Problem.IM1,
        constraints=[MOO_Constraints.IM1_constraint],
        step_size=0.01,
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
        path='/home/kemal/Programming/Python/Articulation/data/pickles/aposteriori/IM1/',
        filename=''
    )
    for i in range(n_tests):
        save_options['filename'] = 'IM1_test_' + str(i+1) + '.pickle'
        params['save_options'] = save_options
        params['seed_value'] = i
        params['test_ID'] = 'IM1_test_' + str(i+1)
        random.seed(i)
        a = random.random()
        params['weights'] = [a, 1-a]
        params['init_sol'] = Solution(np.array([random.uniform(1, 4), random.uniform(1, 2)]))
        SearchInstance = AposterioriWeightingMethod(**params)
        SearchInstance.search()
    
    
def aposteriori_BK1(n_tests=20):
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
    for i in range(n_tests):
        save_options['filename'] = 'BK1_test_' + str(i+1) + '.pickle'
        params['save_options'] = save_options
        params['seed_value'] = i
        params['test_ID'] = 'BK1_test_' + str(i+1)
        random.seed(i)
        a = random.random()
        params['weights'] = [a, 1-a]
        params['init_sol'] = Solution(np.array([random.uniform(-5, 10), random.uniform(-5, 10)]))
        SearchInstance = AposterioriWeightingMethod(**params)
        SearchInstance.search()


if __name__ == '__main__':
    #aposteriori_BK1()
    #aposteriori_IM1(n_tests=20)
    #aposteriori_SCH1()
    aposteriori_FON()


