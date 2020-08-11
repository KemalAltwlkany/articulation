import numpy as np
import sys as sys
import random as random

sys.path.insert(0, "/home/kemal/Programming/Python/Articulation")
from src.PreferenceArticulation.Solution import Solution
from src.PreferenceArticulation.BenchmarkObjectives import *
from src.TabuSearch.progressive_decision_making import ProgressiveArticulationDecisionMaker


def progressive_BK1(n_tests=20):
    DM_params = dict(
        unacceptable_penalty=500,
        unsatisficing_penalty=200,
        narrow_step=5.0,
        max_repetitions=10,
        centroids=[dict(
            a=0,
            b=25,
            c=50,
            d=70,
            e=100
        ),
            dict(
                a=0,
                b=25,
                c=50,
                d=70,
                e=100
            )]
    )
    params = dict(
        init_sol=Solution(np.array([9, 9])),
        problem=MOO_Problem.BK1,
        constraints=[MOO_Constraints.BK1_constraint],
        step_size=0.05,
        neighborhood_size=15,
        max_iter=100,
        M=5000,
        tabu_list_max_length=20,
        max_loops=50,
        search_space_dimensions=2,
        objective_space_dimensions=2,
        save=False
    )
    save_options = dict(
        path='/home/kemal/Programming/Python/Articulation/data/pickles/progressive/BK1/',
        filename=''
    )
    for i in range(n_tests):
        save_options['filename'] = 'BK1_test_' + str(i+1) + '.pickle'
        params['save_options'] = save_options
        params['seed_value'] = i
        params['test_ID'] = 'BK1_test_' + str(i+1)
        random.seed(i)
        np.random.seed(i)
        params['init_sol'] = Solution(np.array([random.uniform(-5, 10), random.uniform(-5, 10)]))
        DMInstance = ProgressiveArticulationDecisionMaker(DM_params, params)
        DMInstance.search()


if __name__ == '__main__':
    progressive_BK1(n_tests=1)

