from src.PreferenceArticulation.ArticulationExceptions import AbstractMethod
import copy as copy
import numpy as np
from src.PreferenceArticulation.Solution import Solution


class TabuSearch:
    def __init__(self, init_sol=None, problem=None, constraints=None, step_size=None, neighborhood_size=None, max_iter=None, M=None,
                 tabu_list_max_length=None, max_loops=None, search_space_dimensions=None, objective_space_dimensions=None):
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
                return self.search_history, "max iter exceeded", it, self.global_best_sol

            # No progress made for max_loops iterations already?
            if it - last_global_sol_improvement > self.max_loops:
                print('Terminating after iteration number ', it, ' because the algorithm hasn''t progressed in ', it - last_global_sol_improvement, ' iterations')
                return self.search_history, 'no progress', it, self.global_best_sol






#
# class TabuSearch(SearchAlgorithm):
#
#     def __init__(self, init_sol=None, problem=None, delta=None, max_iter=None, constraints=None, M=100,
#                  tabu_list_max_length=30, max_loops=30):
#         super().__init__(init_sol=init_sol, problem=problem, delta=delta, max_iter=max_iter,
#                          constraints=constraints, M=M)
#         self.max_loops = max_loops    # maximum steps tolerated for the algorithm to run without any significant change in the search occurring
#         self.tabu_list = []
#         self.tabu_list_max_length = tabu_list_max_length
#         self.global_best_sol = None
#
#     # uses abstract methods, so only child classes can invoke this method
#     def search(self, verbose=False):
#         it = 0
#         # algorithm should evaluate initial solution before anything else
#         self.init_sol.y = self.evaluate_solution(self.init_sol)
#         self.curr_sol = copy.deepcopy(self.init_sol)
#         self.search_history.append(copy.deepcopy(self.curr_sol))
#         iters_no_progress = 0   # number of iterations without any significant change
#         self.global_best_sol = copy.deepcopy(self.curr_sol)
#         while it < self.max_iter:
#
#             it = it + 1
#             prev_sol = copy.deepcopy(self.curr_sol)
#
#             self.generate_neighborhood(self.curr_sol)
#             # Evaluate the individuals now
#             self.evaluate_neighborhood()
#             self.sort_neighborhood()
#
#             # purge neighborhood from Tabu elements
#             while 1:
#                 if self.neighborhood[0] in self.tabu_list:
#                     del self.neighborhood[0]
#                 else:
#                     break
#
#             self.curr_sol = self.neighborhood[0]    # new solution becomes the best of the neighborhood
#             # NEED TO UPDATE GLOBAL SOLUTION!
#
#             if self.__class__.compute_fitness(self.global_best_sol) > self.__class__.compute_fitness(self.curr_sol):
#                 # global solution is worse than newly found solution, update!
#                 self.global_best_sol = copy.deepcopy(self.curr_sol)
#
#             self.search_history.append(copy.deepcopy(self.curr_sol))
#             if prev_sol == self.curr_sol:
#                 print('Terminating after iteration number', it, ' because local extrema was found')
#                 return self.search_history, "local extrema", it
#
#             # remove first 15 elements from tabu list if it is too long
#             # in order to optimize, alg should remove last 14 entries, not first 14
#             if len(self.tabu_list) > self.tabu_list_max_length:
#                 self.tabu_list = self.tabu_list[:-14]
#
#             # in order to optimize, new tabu entries should be put in the beginning, not the end
#             # self.tabu_list.append(copy.deepcopy(self.curr_sol))
#             self.tabu_list.insert(0, copy.deepcopy(self.curr_sol))
#
#             # Check whether search has progressed by a minimum step request
#             # WARNING - CHECK WHETHER THE NEXT LINE WORKS PROPERLY (SHOULD CALL OVERRIDDEN STATIC METHOD)
#             if not self.__class__.progress_measure(prev_sol, self.curr_sol):
#                 iters_no_progress = iters_no_progress + 1
#                 if iters_no_progress > self.max_loops:
#                     print('Terminating after iteration number ', it, ' because the algorithm hasn''t progressed in ', iters_no_progress, ' iterations-')
#                     return self.search_history, 'no progress', it, self.global_best_sol
#             else:
#                 iters_no_progress = 0
#
#
#         print('Terminating because max iterations were exceeded, it = ', it)
#         return self.search_history, "max iter exceeded", it, self.global_best_sol
#
#     # abstract method, must be overridden
#     @staticmethod
#     def progress_measure(sol1, sol2):
#         raise AbstractMethod("Error! Method is abstract and has not been overridden by child class!")
#
#
#     @staticmethod
#     def compute_fitness(sol):
#         raise AbstractMethod("Error! Method is abstract and has not been overriden by child class!")