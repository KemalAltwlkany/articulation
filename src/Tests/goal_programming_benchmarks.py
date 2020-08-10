import numpy as np
import sys as sys
import random as random

sys.path.insert(0, "/home/kemal/Programming/Python/Articulation")
from src.PreferenceArticulation.Solution import Solution
from src.PreferenceArticulation.BenchmarkObjectives import *
from src.TabuSearch.goal_programming import AprioriFuzzyGoalProgramming




def apriori_BK1(n_tests=20):
    # I chose to define 5 different points on the front to which any algorithm should converge.
    # These are an attempt of a uniform distribution along the front.
    aspirations = [[0, 50],  [15, 35], [25, 25], [35, 15], [50, 0]]
    # This means that every 5th test should have the same aspiration level.
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
    for i in range(n_tests):
        save_options['filename'] = 'BK1_test_' + str(i+1) + '.pickle'
        params['save_options'] = save_options
        params['seed_value'] = i
        params['test_ID'] = 'BK1_test_' + str(i+1)
        random.seed(i)
        np.random.seed(i)
        params['aspirations'] = aspirations[i % 5]
        params['init_sol'] = Solution(np.array([random.uniform(-5, 10), random.uniform(-5, 10)]))
        SearchInstance = AprioriFuzzyGoalProgramming(**params)
        SearchInstance.search()


if __name__ == '__main__':
    apriori_BK1(n_tests=100)





