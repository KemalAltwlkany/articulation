from PreferenceArticulation.SearchAlgorithm import SearchAlgorithm
from PreferenceArticulation.ArticulationExceptions import AbstractMethod
import copy as copy

class TabuSearch(SearchAlgorithm):

    def __init__(self, init_sol=None, problem=None, delta=None, max_iter=None, constraints=None, M=100,
                 tabu_list_max_length=30, max_loops=30):
        super().__init__(init_sol=init_sol, problem=problem, delta=delta, max_iter=max_iter,
                         constraints=constraints, M=M)
        self.max_loops = max_loops    # maximum steps tolerated for the algorithm to run without any significant change in the search occurring
        self.tabu_list = []
        self.tabu_list_max_length = tabu_list_max_length
        self.global_best_sol = None

    # uses abstract methods, so only child classes can invoke this method
    def search(self, verbose=False):
        it = 0
        # algorithm should evaluate initial solution before anything else
        self.init_sol.y = self.evaluate_solution(self.init_sol)
        self.curr_sol = copy.deepcopy(self.init_sol)
        self.search_history.append(copy.deepcopy(self.curr_sol))
        iters_no_progress = 0   # number of iterations without any significant change
        self.global_best_sol = copy.deepcopy(self.curr_sol)
        while it < self.max_iter:

            it = it + 1
            prev_sol = copy.deepcopy(self.curr_sol)

            self.generate_neighborhood(self.curr_sol)
            # Evaluate the individuals now
            self.evaluate_neighborhood()
            self.sort_neighborhood()

            # purge neighborhood from Tabu elements
            while 1:
                if self.neighborhood[0] in self.tabu_list:
                    del self.neighborhood[0]
                else:
                    break

            self.curr_sol = self.neighborhood[0]    # new solution becomes the best of the neighborhood
            # NEED TO UPDATE GLOBAL SOLUTION!

            if self.__class__.compute_fitness(self.global_best_sol) > self.__class__.compute_fitness(self.curr_sol):
                # global solution is worse than newly found solution, update!
                self.global_best_sol = copy.deepcopy(self.curr_sol)

            self.search_history.append(copy.deepcopy(self.curr_sol))
            if prev_sol == self.curr_sol:
                print('Terminating after iteration number', it, ' because local extrema was found')
                return self.search_history, "local extrema", it

            # remove first 15 elements from tabu list if it is too long
            # in order to optimize, alg should remove last 14 entries, not first 14
            if len(self.tabu_list) > self.tabu_list_max_length:
                self.tabu_list = self.tabu_list[:-14]

            # in order to optimize, new tabu entries should be put in the beginning, not the end
            # self.tabu_list.append(copy.deepcopy(self.curr_sol))
            self.tabu_list.insert(0, copy.deepcopy(self.curr_sol))

            # Check whether search has progressed by a minimum step request
            # WARNING - CHECK WHETHER THE NEXT LINE WORKS PROPERLY (SHOULD CALL OVERRIDDEN STATIC METHOD)
            if not self.__class__.progress_measure(prev_sol, self.curr_sol):
                iters_no_progress = iters_no_progress + 1
                if iters_no_progress > self.max_loops:
                    print('Terminating after iteration number ', it, ' because the algorithm hasn''t progressed in ', iters_no_progress, ' iterations-')
                    return self.search_history, 'no progress', it, self.global_best_sol
            else:
                iters_no_progress = 0


        print('Terminating because max iterations were exceeded, it = ', it)
        return self.search_history, "max iter exceeded", it, self.global_best_sol

    # abstract method, must be overridden
    @staticmethod
    def progress_measure(sol1, sol2):
        raise AbstractMethod("Error! Method is abstract and has not been overridden by child class!")


    @staticmethod
    def compute_fitness(sol):
        raise AbstractMethod("Error! Method is abstract and has not been overriden by child class!")