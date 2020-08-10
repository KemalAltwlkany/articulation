import numpy as np
import sys as sys
import random as random

sys.path.insert(0, "/home/kemal/Programming/Python/Articulation")
from src.PreferenceArticulation.Solution import Solution
from src.PreferenceArticulation.BenchmarkObjectives import *
from src.TabuSearch.goal_programming import AprioriFuzzyGoalProgramming

def apriori_OSY(n_tests=20):

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

    # I chose to define 5 different points on the front to which any algorithm should converge.
    # These are an attempt of a uniform distribution along the front.
    aspirations = [[-50, 25], [-100, 50], [-175, 75], [-200, 50], [-250, 100]]
    # This means that every group of 20 consecutive tests should have the same aspiration level.
    params = dict(
        init_sol=Solution(np.array([9, 9, 4, 5, 4, 9])),  # unknown whether this solution is feasible.
        problem=MOO_Problem.OSY,
        constraints=[MOO_Constraints.OSY_constraint_1, MOO_Constraints.OSY_constraint_2, MOO_Constraints.OSY_constraint_3,
                     MOO_Constraints.OSY_constraint_4, MOO_Constraints.OSY_constraint_5, MOO_Constraints.OSY_constraint_6,
                     MOO_Constraints.OSY_constraint_7],
        step_size=0.1,
        neighborhood_size=15,
        max_iter=2000,
        M=3000,
        tabu_list_max_length=20,
        max_loops=100,
        search_space_dimensions=6,
        objective_space_dimensions=2,
        save=True,
        aspirations=[0.5, 0.5]
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/apriori/OSY/',
        filename=''
    )
    fifth = n_tests // 5
    which_asp = np.concatenate((np.array([0] * fifth), np.array([1] * fifth), np.array([2] * fifth),
                                np.array([3] * fifth), np.array([4] * fifth)))
    for i in range(n_tests):
        save_options['filename'] = 'OSY_test_' + str(i+1) + '.pickle'
        params['save_options'] = save_options
        params['seed_value'] = i
        params['test_ID'] = 'OSY_test_' + str(i+1)
        random.seed(i)
        np.random.seed(i)
        params['aspirations'] = aspirations[which_asp[i] % 5]
        params['init_sol'] = Solution(generate_feasible_sol())
        SearchInstance = AprioriFuzzyGoalProgramming(**params)
        SearchInstance.search()

def apriori_TNK(n_tests=20):

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

    # I chose to define 5 different points on the front to which any algorithm should converge.
    # These are an attempt of a uniform distribution along the front.
    aspirations = [[1, 0], [0.75, 0.25], [0.5, 0.5], [0.25, 0.75], [0, 1]]
    # This means that every group of 20 consecutive tests should have the same aspiration level.
    params = dict(
        init_sol=Solution(np.array([2, 2])), # unknown whether this solution is feasible.
        problem=MOO_Problem.TNK,
        constraints=[MOO_Constraints.TNK_constraint_1, MOO_Constraints.TNK_constraint_2, MOO_Constraints.TNK_constraint_3],
        step_size=0.05,
        neighborhood_size=15,
        max_iter=2500,
        M=100,
        tabu_list_max_length=20,
        max_loops=200,
        search_space_dimensions=2,
        objective_space_dimensions=2,
        save=True,
        aspirations=[0.5, 0.5]
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/apriori/TNK/',
        filename=''
    )
    fifth = n_tests // 5
    which_asp = np.concatenate((np.array([0] * fifth), np.array([1] * fifth), np.array([2] * fifth),
                                np.array([3] * fifth), np.array([4] * fifth)))
    for i in range(n_tests):
        save_options['filename'] = 'TNK_test_' + str(i+1) + '.pickle'
        params['save_options'] = save_options
        params['seed_value'] = i
        params['test_ID'] = 'TNK_test_' + str(i+1)
        random.seed(i)
        np.random.seed(i)
        params['aspirations'] = aspirations[which_asp[i] % 5]
        params['init_sol'] = Solution(generate_feasible_sol())
        SearchInstance = AprioriFuzzyGoalProgramming(**params)
        SearchInstance.search()

def apriori_FON(n_tests=20, n_dims=2):
    # I chose to define 5 different points on the front to which any algorithm should converge.
    # These are an attempt of a uniform distribution along the front.
    f1 = [0.9457533241109305, 0.8399240327613827, 0.6321205588285577, 0.34156746712171393, 0.08220978425157577]
    aspirations = [[f1[i], f1[-i-1]] for i in range(5)]
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
        aspirations=[1./n_dims] * 2
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
        params['aspirations'] = aspirations[which_asp[i] % 5]
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
    #apriori_SCH1(n_tests=100)
    #apriori_FON(n_tests=100, n_dims=5)
    #apriori_TNK(n_tests=10)
    apriori_OSY(n_tests=100)



