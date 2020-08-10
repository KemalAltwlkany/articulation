import numpy as np
import sys as sys
import random as random

sys.path.insert(0, "/home/kemal/Programming/Python/Articulation")
from src.PreferenceArticulation.Solution import Solution
from src.PreferenceArticulation.BenchmarkObjectives import *
from src.TabuSearch.goal_programming import AprioriFuzzyGoalProgramming



def apriori_FON(n_tests=20, n_dims=2):
    # I chose to define 5 different points on the front to which any algorithm should converge.
    # These are an attempt of a uniform distribution along the front.
    aspirations = [[0, 4], [0.25, 2.25], [1, 1], [2.25, 0.25], [4, 0]]
    # This means that every group of 20 consecutive tests should have the same aspiration level.
    params = dict(
        init_sol=Solution(np.array([3.5]*n_dims)),
        problem=MOO_Problem.FON,
        constraints=[],  # no constraints for this problem
        step_size=0.05,
        neighborhood_size=20,
        max_iter=4000,
        M=100,
        tabu_list_max_length=60,
        max_loops=300,
        search_space_dimensions=n_dims,
        objective_space_dimensions=2,
        save=True,
        aspirations=[1./n_dims] * n_dims
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/apriori/FON/',
        filename=''
    )
    fifth = n_tests // 5
    which_asp = np.concatenate((np.array([0] * fifth), np.array([1] * fifth), np.array([2] * fifth),
                                np.array([3] * fifth), np.array([4] * fifth)))
    for i in range(n_tests):
        save_options['filename'] = 'FON_test_' + str(i+1) + '.pickle'
        params['save_options'] = save_options
        params['seed_value'] = i
        params['test_ID'] = 'FON_test_' + str(i+1)
        random.seed(i)
        np.random.seed(i)

        params['init_sol'] = Solution(np.random.uniform(low=-4, high=4, size=n_dims))
        SearchInstance = AprioriFuzzyGoalProgramming(**params)
        SearchInstance.search()

def apriori_SCH1(n_tests=20):
    # I chose to define 5 different points on the front to which any algorithm should converge.
    # These are an attempt of a uniform distribution along the front.
    aspirations = [[0, 4], [0.25, 2.25], [1, 1], [2.25, 0.25], [4, 0]]
    # This means that every group of 20 consecutive tests should have the same aspiration level.
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
        aspirations=[0.5, 0.5]
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/apriori/SCH1/',
        filename=''
    )
    fifth = n_tests // 5
    which_asp = np.concatenate((np.array([0] * fifth), np.array([1] * fifth), np.array([2] * fifth), np.array([3] * fifth), np.array([4] * fifth)))
    for i in range(n_tests):
        save_options['filename'] = 'SCH1_test_' + str(i+1) + '.pickle'
        params['save_options'] = save_options
        params['seed_value'] = i
        params['test_ID'] = 'SCH1_test_' + str(i+1)
        random.seed(i)
        np.random.seed(i)
        a = random.random()
        params['aspirations'] = aspirations[which_asp[i] % 5]
        # Since the search space is 1D, pick initial point as either -10 or 10.
        if i % 2 is 0:
            params['init_sol'] = Solution(np.array([-5 + random.uniform(-1, 1)]))
        else:
            params['init_sol'] = Solution(np.array([5 + random.uniform(-1, 1)]))
        SearchInstance = AprioriFuzzyGoalProgramming(**params)
        SearchInstance.search()

def apriori_IM1(n_tests=20):
    # I chose to define 5 different points on the front to which any algorithm should converge.
    # These are an attempt of a uniform distribution along the front.
    aspirations = [[4, 1], [3.5, 1.9375], [3, 2.75], [2.5, 3.4375], [2, 4]]
    # This means that every group of 20 consecutive tests should have the same aspiration level.

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
        aspirations=[0.5, 0.5]
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/apriori/IM1/',
        filename=''
    )
    fifth = n_tests//5
    which_asp = np.concatenate((np.array([0] * fifth), np.array([1] * fifth), np.array([2] * fifth), np.array([3] * fifth), np.array([4] * fifth)))
    for i in range(n_tests):
        save_options['filename'] = 'IM1_test_' + str(i+1) + '.pickle'
        params['save_options'] = save_options
        params['seed_value'] = i
        params['test_ID'] = 'IM1_test_' + str(i+1)
        random.seed(i)
        np.random.seed(i)
        params['aspirations'] = aspirations[which_asp[i] % 5]  # 5 different types of aspiration levels
        params['init_sol'] = Solution(np.array([random.uniform(1, 4), random.uniform(1, 2)]))
        SearchInstance = AprioriFuzzyGoalProgramming(**params)
        SearchInstance.search()

def apriori_BK1(n_tests=20):
    # I chose to define 5 different points on the front to which any algorithm should converge.
    # These are an attempt of a uniform distribution along the front.
    aspirations = [[0, 50],  [15, 35], [25, 25], [35, 15], [50, 0]]
    # This means that every group of 20 consecutive tests should have the same aspiration level.
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
        aspirations=[0, 30]
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/apriori/BK1/',
        filename=''
    )
    fifth = n_tests//5
    which_asp = np.concatenate((np.array([0] * fifth), np.array([1] * fifth), np.array([2] * fifth), np.array([3] * fifth), np.array([4] * fifth)))
    for i in range(n_tests):
        save_options['filename'] = 'BK1_test_' + str(i+1) + '.pickle'
        params['save_options'] = save_options
        params['seed_value'] = i
        params['test_ID'] = 'BK1_test_' + str(i+1)
        random.seed(i)
        np.random.seed(i)
        params['aspirations'] = aspirations[which_asp[i] % 5]  # 5 different types of aspiration levels
        params['init_sol'] = Solution(np.array([random.uniform(-5, 10), random.uniform(-5, 10)]))
        SearchInstance = AprioriFuzzyGoalProgramming(**params)
        SearchInstance.search()


if __name__ == '__main__':
    #apriori_BK1(n_tests=100)
    #apriori_IM1(n_tests=100)
    apriori_SCH1(n_tests=10)





