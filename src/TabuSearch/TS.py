from src.PreferenceArticulation.ArticulationExceptions import AbstractMethod
import copy as copy
import numpy as np
from src.PreferenceArticulation.Solution import Solution
import os as os
import pickle as pickle


class TabuSearch:
    def __init__(self, init_sol=None, problem=None, constraints=None, step_size=None, neighborhood_size=None, max_iter=None, M=None,
                 tabu_list_max_length=None, max_loops=None, search_space_dimensions=None, objective_space_dimensions=None, save=False,
                 save_options=None):
        # attributes necessary for the actual search
        self.init_sol = init_sol
        self.curr_sol = None
        self.global_best_sol = None

        self.tabu_list = []
        self.TL_max_length = tabu_list_max_length

        self.step_size = step_size
        self.neighborhood_x_vector = None
        self.neighborhood = None
        self.neighborhood_size = neighborhood_size

        self.max_iter = max_iter
        self.max_loops = max_loops
        self.m = objective_space_dimensions
        self.n = search_space_dimensions

        # attributes regarding the MOO aspect of the problem
        self.problem = problem  # static method of class MOO_Problem
        self.constraints = constraints  # list of constraints
        self.M = M  # criterion deterioration factor for each constraint violation

        # attributes of interest for analyzing the search process, i.e. not actually required for optimizing a problem
        self.search_history = None
        self.save = save
        self.save_options = save_options

    # abstract
    def evaluate_solution(self, sol):
        raise AbstractMethod("Error! Method is abstract and has not been overridden by child class!")

    def evaluate_objectives(self, sol):
        sol.set_y(self.problem(sol.x))

    def penalty(self, sol):
        penalty_value = 0
        for constraint in self.constraints:
            penalty_value = penalty_value + constraint(sol.x)
        return penalty_value * self.M

    def generate_neighborhood_x_vectors(self):
        """
        Generating neighborhooding solutions is done using as per:
        Baykasoglu, Adil. "Applying multiple objective tabu search to continuous optimization problems with a simple neighbourhood strategy." International Journal for Numerical Methods in Engineering 65.3 (2006): 406-424.
        Chelouah, Rachid, and Patrick Siarry. "Tabu search applied to global optimization." European journal of operational research 123.2 (2000): 256-270.
        :return:
        """
        # Create a n_size x n matrix. This is an array of random values to be added.
        self.neighborhood_x_vector = (self.step_size * np.random.random_sample((self.neighborhood_size, self.n)) - self.step_size * 0.5)

        # Add current sols to the neighborhood matrix. Save results in neighborhood matrix.
        np.add(self.neighborhood_x_vector, self.curr_sol.get_x(), self.neighborhood_x_vector)

    def search(self):
        # Evaluate initial solution.
        self.curr_sol = copy.deepcopy(self.init_sol)
        self.evaluate_objectives(self.curr_sol)
        self.evaluate_solution(self.curr_sol)
        self.global_best_sol = copy.deepcopy(self.curr_sol)

        self.search_history = []
        it = 0
        prev_sol = None
        last_global_sol_improvement = 0
        while 1:
            self.search_history.append(copy.deepcopy(self.curr_sol))
            prev_sol = copy.deepcopy(self.curr_sol)
            self.generate_neighborhood_x_vectors()
            self.neighborhood = [Solution(self.neighborhood_x_vector[i]) for i in range(self.neighborhood_size)]
            for sol in self.neighborhood:
                self.evaluate_objectives(sol)
                self.evaluate_solution(sol)

            self.neighborhood.sort(key=Solution.get_val)

            # purge neighborhood from Tabu elements
            # I did not cover the case where theoretically every element from the neighborhood is in the tabu list.
            while 1:
                if self.neighborhood[0] in self.tabu_list:
                    del self.neighborhood[0]
                else:
                    break

            # Update current solution
            self.curr_sol = copy.deepcopy(self.neighborhood[0])

            # Update global best solution if necessary
            if self.global_best_sol.get_val() > self.curr_sol.get_val():
                self.global_best_sol = copy.deepcopy(self.curr_sol)
                last_global_sol_improvement = it


            # Update tabu-list
            # As an optimization trick, remove the 5 first entries in the TL, instead of just the oldest element.
            if len(self.tabu_list) > self.TL_max_length:
                self.tabu_list = self.tabu_list[5:]
            self.tabu_list.append(copy.deepcopy(prev_sol))


            it = it + 1
            # Check termination conditions:

            # Maximum iterations exceeded?
            if it > self.max_iter:
                print('Terminating because max iterations were exceeded, it = ', it)
                if self.save is True:
                    self.save_search_results()
                return_dict = dict(search_history=self.search_history, termination_reason='max iter exceeded', last_iter=it, global_best_sol=self.global_best_sol)
                return return_dict

            # No progress made for max_loops iterations already?
            if it - last_global_sol_improvement > self.max_loops:
                print('Terminating after iteration number ', it, ' because the algorithm hasn''t progressed in ', it - last_global_sol_improvement, ' iterations')
                if self.save is True:
                    self.save_search_results()
                return_dict = dict(search_history=self.search_history, termination_reason='no progress', last_iter=it, global_best_sol=self.global_best_sol)
                return return_dict

    def save_search_results(self):
        # assumes self.save_options is a dictionary containing all necessary keys/vals
        # memorize cwd
        cwd = os.getcwd()
        # switch to save folder location
        os.chdir(self.save_options['path'])
        # I can literally save the entire instance to a dictionary and pickle the dictionary
        all_info = self.__dict__
        with open(self.save_options['filename'], 'wb') as f:
            pickle.dump(all_info, f, pickle.HIGHEST_PROTOCOL)
        os.chdir(cwd)






