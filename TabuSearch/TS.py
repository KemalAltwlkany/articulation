from PreferenceArticulation.SearchAlgorithm import SearchAlgorithm
from PreferenceArticulation.ArticulationExceptions import AbstractMethod
import copy as copy
import time as time

class TabuSearch(SearchAlgorithm):

    def __init__(self, init_sol=None, problem=None, delta=None, max_iter=None, constraints=None, M=100,
                 tabu_list_max_length=30, max_loops=30):
        super().__init__(init_sol=init_sol, problem=problem, delta=delta, max_iter=max_iter,
                         constraints=constraints, M=M)
        self.max_loops = max_loops    # maximum steps tolerated for the algorithm to run without any significant change in the search occurring
        self.tabu_list = []
        self.tabu_list_max_length = tabu_list_max_length

    # uses abstract methods, so only child classes can invoke this method
    def search(self, verbose=False):
        it = 0
        # algorithm should evaluate initial solution before anything else
        self.init_sol.y = self.evaluate_solution(self.init_sol)
        self.curr_sol = copy.deepcopy(self.init_sol)
        self.search_history.append(copy.deepcopy(self.curr_sol))
        iters_no_progress = 0   # number of iterations without any significant change
        all_iters_time = []
        while it < self.max_iter:
            total_time = time.time()
            s1 = time.time()
            it = it + 1
            prev_sol = copy.deepcopy(self.curr_sol)
            s2 = time.time()
            print("Time needed for iter increment + copy.deepcopy is: ", s2 - s1)

            s1 = time.time()
            self.generate_neighborhood(self.curr_sol)
            s2 = time.time()
            print("Time needed for generate neighborhood is : ", s2 - s1)

            # Evaluating the fitness of a solution is costly. It is better to firstly purge the Tabu list
            # and to evaluate the neighborhood afterwards
            # Prone to mistakes. Should verify.
            # s1 = time.time()
            # for tabu_sol in self.tabu_list:
            #     if tabu_sol in self.neighborhood:
            #         self.neighborhood.remove(tabu_sol)
            # s2 = time.time()
            # print("Time needed for purging tabu list before evaluating solutions: ", s2 - s1)


            # It turned out questioning the statement above was good. It is not worth it to firstly purge the
            # Tabu list and to evaluate the neighborhood afterwards. It is better to evaluate the neighborhood,
            # and to continue picking solutions until they are no longer in the tabu list. Logically, we should start
            # checking from the end of the tabu list (if the elements are there).



            # Evaluate the individuals now
            s1 = time.time()
            self.evaluate_neighborhood()
            s2 = time.time()
            print("Time needed for evaluating neighborhood:", s2 - s1)

            s1 = time.time()
            self.sort_neighborhood()
            s2 = time.time()
            print("Time needed for sorting neighborhood: ", s2 - s1)

            s1 = time.time()
            while 1:
                if self.neighborhood[0] in self.tabu_list:
                    del self.neighborhood[0]
                else:
                    break
            s2 = time.time()
            print("Time needed for purging neighborhood from taboos: ", s2 - s1)

            s1 = time.time()
            self.curr_sol = self.neighborhood[0]    # new solution becomes the best of the neighborhood
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
            s2 = time.time()
            print("Time needed for updating tabu list and curr solution: ", s2 - s1)


            s1 = time.time()
            # Check whether search has progressed by a minimum step request
            # WARNING - CHECK WHETHER THE NEXT LINE WORKS PROPERLY (SHOULD CALL OVERRIDDEN STATIC METHOD)
            if not self.__class__.progress_measure(prev_sol, self.curr_sol):
                iters_no_progress = iters_no_progress + 1
                if iters_no_progress > self.max_loops:
                    print('Terminating after iteration number ', it, ' because the algorithm hasn''t progressed in ', iters_no_progress, ' iterations-')
                    return self.search_history, 'no progress', it
            else:
                iters_no_progress = 0
            s2 = time.time()
            print("Time needed for checking whether search stagnates: ", s2 - s1)

            total_time_2 = time.time()
            print("TOTAL ITERATION TIME: ", total_time_2 -total_time)
            all_iters_time.append(total_time_2-total_time)
            print("-------------------------------------------------------------------------")
            if verbose is True:
                print("----------------------------------------------------------")
                print("Iteration number = ", it)
                print("Previous solution, decision variables = ", ["%.3f" % i for i in prev_sol.x])
                print("Previous solution, objectives = ", ["%.3f" % i for i in prev_sol.y])
                print("Current solution, decision variables = ", ["%.3f" % i for i in self.curr_sol.x])
                print("Current solution, objectives = ", ["%.3f" %i for i in self.curr_sol.y])
                # print("First 5 elements of Tabu list = ",  self.tabu_list[:5])
                # print("Last 5 elements of Tabu list = ",  self.tabu_list[-5:])
                print("----------------------------------------------------------")

        print('Terminating because max iterations were exceeded, it = ', it)
        print('AVERAGE ITERATION TIME: ', sum(all_iters_time)/len(all_iters_time))
        return self.search_history, "max iter exceeded", it

    # abstract method, must be overridden
    @staticmethod
    def progress_measure(sol1, sol2):
        raise AbstractMethod("Error! Method is abstract and has not been overridden by child class!")


