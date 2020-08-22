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

def progressive_FON(n_tests=20, n_repetitions=10, n_dims=5):
    search_params = dict(
        init_sol=Solution(np.array([3.5] * n_dims)),
        problem=MOO_Problem.FON,
        constraints=[MOO_Constraints.FON_constraint],  # no constraints for this problem
        step_size=0.02,
        neighborhood_size=40,
        max_iter=200,
        M=5000,  # must be a greater value for progressive art.
        tabu_list_max_length=60,
        max_loops=70,  # must be a lesser value for progressive art.
        search_space_dimensions=n_dims,
        objective_space_dimensions=2,
        save=False,  # important!
        dynamic_constraints=None,
        aspirations=[0, 0]
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/progressive/FON/',
        filename=''
    )
    IDM_params = dict(
        problem_name='FON',
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
        save_options['filename'] = 'FON_test_' + str(i + 1) + '.pickle'
        search_params['save_options'] = save_options
        IDM_params['save_options'] = save_options
        IDM_params['which_csv_index'] = str(which_csv[i])
        search_params['seed_value'] = i
        search_params['test_ID'] = 'FON_test_' + str(i + 1)
        random.seed(i)
        np.random.seed(i)
        search_params['init_sol'] = Solution(np.random.uniform(low=-4, high=4, size=n_dims))

        AgentInstance = IntelligentDM(IDM_params, search_params)
        AgentInstance.procedure()

def progressive_TNK(n_tests=20, n_repetitions=10):

    # Copied from goal programming
    def generate_feasible_sol():
        while True:
            x1 = random.uniform(0, math.pi)
            x2 = random.uniform(0, math.pi)
            if math.isclose(x1, 0):
                c1 = math.pow(x2, 2) - 1.1
            else:
                c1 = math.pow(x1, 2) + math.pow(x2, 2) - 1 - 0.1 * math.cos(16 * math.atan(x2 / x1))
            c2 = math.pow(x1 - 0.5, 2) + math.pow(x2 - 0.5, 2) - 0.5
            if c1 >= 0 >= c2:
                return np.array([x1, x2])

    search_params = dict(
        init_sol=Solution(np.array([2, 2])),  # unknown whether this solution is feasible
        problem=MOO_Problem.TNK,
        constraints=[MOO_Constraints.TNK_constraint_1, MOO_Constraints.TNK_constraint_2, MOO_Constraints.TNK_constraint_3],  # no constraints for this problem
        step_size=0.05,
        neighborhood_size=15,
        max_iter=150,
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
        path='/home/kemal/Programming/Python/Articulation/data/pickles/progressive/TNK/',
        filename=''
    )
    IDM_params = dict(
        problem_name='TNK',
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
        save_options['filename'] = 'TNK_test_' + str(i + 1) + '.pickle'
        search_params['save_options'] = save_options
        IDM_params['save_options'] = save_options
        IDM_params['which_csv_index'] = str(which_csv[i])
        search_params['seed_value'] = i
        search_params['test_ID'] = 'TNK_test_' + str(i + 1)
        random.seed(i)
        np.random.seed(i)
        search_params['init_sol'] = Solution(generate_feasible_sol())
        AgentInstance = IntelligentDM(IDM_params, search_params)
        AgentInstance.procedure()

def progressive_OSY(n_tests=20, n_repetitions=10):

    # Copied from goal programming
    def generate_feasible_sol():
        while True:  # 5k sample points
            x1 = random.uniform(0, 10)
            x2 = random.uniform(0, 10)
            x3 = random.uniform(1, 5)
            x4 = random.uniform(0, 6)
            x5 = random.uniform(1, 5)
            x6 = random.uniform(0, 10)
            if x1 + x2 - 2 < 0:
                continue
            if 6 - x1 - x2 < 0:
                continue
            if 2 - x2 + x1 < 0:
                continue
            if 2 - x1 + 3 * x2 < 0:
                continue
            if 4 - (x3 - 3) ** 2 - x4 < 0:
                continue
            if (x5 - 3) ** 2 + x6 - 4 < 0:
                continue
            return np.array([x1, x2, x3, x4, x5, x6])

    search_params = dict(
        init_sol=Solution(np.array([9, 9, 4, 5, 4, 9])),  # unknown whether this solution is feasible
        problem=MOO_Problem.OSY,
        constraints=[MOO_Constraints.OSY_constraint_1, MOO_Constraints.OSY_constraint_2, MOO_Constraints.OSY_constraint_3,
                     MOO_Constraints.OSY_constraint_4, MOO_Constraints.OSY_constraint_5, MOO_Constraints.OSY_constraint_6,
                     MOO_Constraints.OSY_constraint_7],
        step_size=0.1,
        neighborhood_size=15,
        max_iter=300,
        M=5000,  # must be a greater value for progressive art.
        tabu_list_max_length=20,
        max_loops=80,  # must be a lesser value for progressive art.
        search_space_dimensions=6,
        objective_space_dimensions=2,
        save=False,  # important!
        dynamic_constraints=None,
        aspirations=[0, 0]
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/progressive/OSY/',
        filename=''
    )
    IDM_params = dict(
        problem_name='OSY',
        pen_val=1000,  # must be greater for OSY
        n_repetitions=n_repetitions,
        save=True,
        save_options=None,
        which_csv_index='0'
    )
    fifth = n_tests // 5
    which_csv = np.concatenate((np.array([0] * fifth), np.array([1] * fifth), np.array([2] * fifth),
                                np.array([3] * fifth), np.array([4] * fifth)))
    for i in range(n_tests):
        save_options['filename'] = 'OSY_test_' + str(i + 1) + '.pickle'
        search_params['save_options'] = save_options
        IDM_params['save_options'] = save_options
        IDM_params['which_csv_index'] = str(which_csv[i])
        search_params['seed_value'] = i
        search_params['test_ID'] = 'OSY_test_' + str(i + 1)
        random.seed(i)
        np.random.seed(i)
        search_params['init_sol'] = Solution(generate_feasible_sol())
        AgentInstance = IntelligentDM(IDM_params, search_params)
        AgentInstance.procedure()



if __name__ == '__main__':
    #progressive_BK1(n_tests=200, n_repetitions=10)
    #progressive_IM1(n_tests=200, n_repetitions=10)
    #progressive_SCH1(n_tests=200, n_repetitions=10)
    #progressive_FON(n_tests=200, n_repetitions=10, n_dims=5)
    #progressive_TNK(n_tests=200, n_repetitions=10)
    progressive_OSY(n_tests=200, n_repetitions=10)


