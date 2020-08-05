from src.PreferenceArticulation.ArticulationExceptions import AbstractMethod
import copy as copy
import numpy as np
from src.PreferenceArticulation.Solution import Solution
from src.TabuSearch.TS import TabuSearch


class AposterioriWeightingMethod(TabuSearch):


    def __init__(self, init_sol=None, problem=None, constraints=None, step_size=None, neighborhood_size=None,
                 max_iter=None, M=None, tabu_list_max_length=None, max_loops=None, search_space_dimensions=None,
                 objective_space_dimensions=None, save=False, save_options=None, weights=None):
        super().__init__(init_sol, problem, constraints, step_size, neighborhood_size, max_iter, M,
                         tabu_list_max_length, max_loops, search_space_dimensions, objective_space_dimensions, save,
                         save_options)
        self.weights = np.array(weights)
        # as an optimization enhancement, precompute the sum of weights
        self.sum_of_weights = np.sum(self.weights)


    def evaluate_solution(self, sol):
        # In the weighting method, the solution is evaluated by creating a weighted sum of all objective values
        sol.set_val(np.average(sol.get_y(), weights=self.weights)*self.sum_of_weights - self.penalty(sol))







