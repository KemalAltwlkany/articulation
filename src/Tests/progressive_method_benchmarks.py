import numpy as np
import random as random

from src.PreferenceArticulation.Solution import Solution
from src.PreferenceArticulation.BenchmarkObjectives import *
from src.TabuSearch.progressive_method import IntelligentDM

def progressive_BK1(n_tests=20, n_repetitions=10):
    search_params = dict(
        init_sol=Solution(np.array([9, 9])),
        problem=MOO_Problem.BK1,
        constraints=[MOO_Constraints.BK1_constraint],
        step_size=0.05,
        neighborhood_size=15,
        max_iter=2000,
        M=5000,  # must be a greater value for progressive art.
        tabu_list_max_length=20,
        max_loops=50,  # must be a lesser value for progressive art.
        search_space_dimensions=2,
        objective_space_dimensions=2,
        save=False,  # important!
        dynamic_constraints=None
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
        save_options=None
    )
    for i in range(n_tests):
        save_options['filename'] = 'BK1_test_' + str(i + 1) + '.pickle'
        search_params['save_options'] = save_options
        IDM_params['save_options'] = save_options
        search_params['seed_value'] = i
        search_params['test_ID'] = 'BK1_test_' + str(i + 1)
        random.seed(i)
        np.random.seed(i)
        search_params['init_sol'] = Solution(np.array([random.uniform(-5, 10), random.uniform(-5, 10)]))
        AgentInstance = IntelligentDM(IDM_params, search_params)
        AgentInstance.procedure()






