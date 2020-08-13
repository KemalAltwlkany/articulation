import numpy as np
import random as random
import copy as copy

from src.PreferenceArticulation.Solution import Solution
from src.PreferenceArticulation.BenchmarkObjectives import *
from src.TabuSearch.progressive_method import IntelligentDM

# Unlike in a priori articulation where the aspiration levels are modified through a for-loop, here
# the a priori information is needed through 5 variables, so instead of modifying it in a loop, I'll simply be passing
# an argument indicating which .csv file containing articulated data should be used.
# these .csv files are used for training the RBDTree.
def progressive_BK1(n_tests=20, n_repetitions=10):
    search_params = dict(
        init_sol=Solution(np.array([9, 9])),
        problem=MOO_Problem.BK1,
        constraints=[MOO_Constraints.BK1_constraint],
        step_size=0.05,
        neighborhood_size=15,
        max_iter=100,
        M=5000,  # must be a greater value for progressive art.
        tabu_list_max_length=20,
        max_loops=50,  # must be a lesser value for progressive art.
        search_space_dimensions=2,
        objective_space_dimensions=2,
        save=False,  # important!
        dynamic_constraints=None,
        aspirations=[0, 0]
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/progressive/BK1/',
        filename=''
    )
    IDM_params = dict(
        problem_name='BK1',
        pen_val=500,
        n_repetitions=n_repetitions,
        save=True,
        save_options=None,
        which_csv_index='0'
    )
    fifth = n_tests // 5
    which_csv = np.concatenate((np.array([0] * fifth), np.array([1] * fifth), np.array([2] * fifth),
                                np.array([3] * fifth), np.array([4] * fifth)))
    for i in range(n_tests):
        save_options['filename'] = 'BK1_test_' + str(i + 1) + '.pickle'
        search_params['save_options'] = save_options
        IDM_params['save_options'] = save_options
        IDM_params['which_csv_index'] = str(which_csv[i])
        search_params['seed_value'] = i
        search_params['test_ID'] = 'BK1_test_' + str(i + 1)
        random.seed(i)
        np.random.seed(i)
        search_params['init_sol'] = Solution(np.array([random.uniform(-5, 10), random.uniform(-5, 10)]))
        AgentInstance = IntelligentDM(IDM_params, search_params)
        AgentInstance.procedure()

def progressive_IM1(n_tests=20, n_repetitions=10):
    search_params = dict(
        init_sol=Solution(np.array([2, 1.5])),
        problem=MOO_Problem.IM1,
        constraints=[MOO_Constraints.IM1_constraint],
        step_size=0.01,
        neighborhood_size=15,
        max_iter=100,
        M=5000,  # must be a greater value for progressive art.
        tabu_list_max_length=20,
        max_loops=50,  # must be a lesser value for progressive art.
        search_space_dimensions=2,
        objective_space_dimensions=2,
        save=False,  # important!
        dynamic_constraints=None,
        aspirations=[0, 0]
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/progressive/IM1/',
        filename=''
    )
    IDM_params = dict(
        problem_name='IM1',
        pen_val=500,
        n_repetitions=n_repetitions,
        save=True,
        save_options=None,
        which_csv_index='0'
    )
    fifth = n_tests // 5
    which_csv = np.concatenate((np.array([0] * fifth), np.array([1] * fifth), np.array([2] * fifth),
                                np.array([3] * fifth), np.array([4] * fifth)))
    for i in range(n_tests):
        save_options['filename'] = 'IM1_test_' + str(i + 1) + '.pickle'
        search_params['save_options'] = save_options
        IDM_params['save_options'] = save_options
        IDM_params['which_csv_index'] = str(which_csv[i])
        search_params['seed_value'] = i
        search_params['test_ID'] = 'IM1_test_' + str(i + 1)
        random.seed(i)
        np.random.seed(i)
        search_params['init_sol'] = Solution(np.array([random.uniform(1, 4), random.uniform(1, 2)]))
        AgentInstance = IntelligentDM(IDM_params, search_params)
        AgentInstance.procedure()

def progressive_SCH1(n_tests=20, n_repetitions=10):
    search_params = dict(
        init_sol=Solution(np.array([10])),
        problem=MOO_Problem.SCH1,
        constraints=[],  # no constraints for this problem
        step_size=0.02,
        neighborhood_size=15,
        max_iter=100,
        M=5000,  # must be a greater value for progressive art.
        tabu_list_max_length=20,
        max_loops=50,  # must be a lesser value for progressive art.
        search_space_dimensions=1,
        objective_space_dimensions=2,
        save=False,  # important!
        dynamic_constraints=None,
        aspirations=[0, 0]
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/progressive/SCH1/',
        filename=''
    )
    IDM_params = dict(
        problem_name='IM1',
        pen_val=500,
        n_repetitions=n_repetitions,
        save=True,
        save_options=None,
        which_csv_index='0'
    )
    fifth = n_tests // 5
    which_csv = np.concatenate((np.array([0] * fifth), np.array([1] * fifth), np.array([2] * fifth),
                                np.array([3] * fifth), np.array([4] * fifth)))
    for i in range(n_tests):
        save_options['filename'] = 'SCH1_test_' + str(i + 1) + '.pickle'
        search_params['save_options'] = save_options
        IDM_params['save_options'] = save_options
        IDM_params['which_csv_index'] = str(which_csv[i])
        search_params['seed_value'] = i
        search_params['test_ID'] = 'SCH1_test_' + str(i + 1)
        random.seed(i)
        np.random.seed(i)
        # Since the search space is 1D, pick initial point as either -10 or 10.
        if i % 2 is 0:
            search_params['init_sol'] = Solution(np.array([-5 + random.uniform(-1, 1)]))
        else:
            search_params['init_sol'] = Solution(np.array([5 + random.uniform(-1, 1)]))
        AgentInstance = IntelligentDM(IDM_params, search_params)
        AgentInstance.procedure()



if __name__ == '__main__':
    #progressive_BK1(n_tests=100, n_repetitions=10)
    #progressive_IM1(n_tests=10, n_repetitions=10)
    progressive_SCH1(n_tests=10, n_repetitions=10)




