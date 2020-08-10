import numpy as np
from src.TabuSearch.TS import TabuSearch

# Fuzzy goal programming, or min-max goal programming
class AprioriFuzzyGoalProgramming(TabuSearch):

    def __init__(self, init_sol=None, problem=None, constraints=None, step_size=None, neighborhood_size=None,
                 max_iter=None, M=None, tabu_list_max_length=None, max_loops=None, search_space_dimensions=None,
                 objective_space_dimensions=None, save=False, save_options=None, seed_value=0, test_ID=None, aspirations=None):
        super().__init__(init_sol, problem, constraints, step_size, neighborhood_size, max_iter, M,
                         tabu_list_max_length, max_loops, search_space_dimensions, objective_space_dimensions, save,
                         save_options, seed_value, test_ID)

        self.aspirations = np.array(aspirations)

    def evaluate_solution(self, sol):
        """
        In fuzzy goal programming, the single criterion function which is to be minimized is:
        f = minimize max {d_{j}}
        where j=1, 2,..., m
        d_j is the deviation of the j-th objective from the j-th aspiration level.
        Basically, minimize the current largest deviation from any goal.
        :param sol:
        :return:
        """
        sol.set_val(np.max(np.subtract(sol.get_y(), self.aspirations)) + self.penalty(sol))





