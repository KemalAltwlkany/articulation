from TabuSearch.TS import TabuSearch
import math as math


class TabuSearchApriori(TabuSearch):

    weights = [1, 1]
    n_objectives = 2
    min_progress = 0.001

    def __init__(self, init_sol=None, problem=None, delta=None, max_iter=None, constraints=None, M=100,
                 tabu_list_max_length=30, weights=None, n_objectives=None, max_loops=30, min_progress=0.001):
        super().__init__(init_sol=init_sol, problem=problem, delta=delta, max_iter=max_iter, constraints=constraints,
                         M=M, tabu_list_max_length=tabu_list_max_length, max_loops=max_loops)
        TabuSearchApriori.weights = weights
        TabuSearchApriori.n_objectives = n_objectives
        TabuSearchApriori.min_progress = min_progress

    def sort_neighborhood(self):
        self.neighborhood.sort(key=TabuSearchApriori.compute_fitness)

    @staticmethod
    def compute_fitness(sol):
        """
        In apriori articulation, a full order can be introduced within the set of solutions being considered.
        The parameter/key which is used to establish full order is called the fitness of the solution.
        :param sol:
        :return:
        """
        fit = 0
        for i in range(TabuSearchApriori.n_objectives):
            fit = fit + TabuSearchApriori.weights[i] * sol.y[i]
        return fit

    @staticmethod
    def progress_measure(sol1, sol2):
        """
        For two solutions, run through their objective vectors. If there is no significant progress, return False.
        If at least one objective differs from the other solution's objective by class static parameter min_progress
        then return True, as there exists a progress in the search algorithm.
        :param sol1: class Solution
        :param sol2: class Solution
        :return: boolean, True if there exists progress, False if there is no progress
        """
        for fi, fj in zip(sol1.y, sol2.y):
            if not math.isclose(fi, fj, rel_tol=TabuSearchApriori.min_progress, abs_tol=TabuSearchApriori.min_progress):
                return True
        return False
