import copy as copy
from PreferenceArticulation.SearchAlgorithm import SearchAlgorithm


class LocalSearch(SearchAlgorithm):

    def __init__(self, init_sol=None, problem =None, delta=None, max_iter=None, constraints=None, M=100):
        super().__init__(init_sol=init_sol, problem=problem, delta=delta, max_iter=max_iter, constraints=constraints, M=M)

    # uses abstract methods, so only child classes can invoke this method
    def search(self):
        it = 0
        # algorithm should evaluate initial solution before anything else
        self.init_sol.y = self.evaluate_solution(self.init_sol)
        self.curr_sol = copy.deepcopy(self.init_sol)
        self.search_history.append(copy.deepcopy(self.curr_sol))
        while it < self.max_iter:
            it = it + 1
            prev_sol = copy.deepcopy(self.curr_sol)

            self.generate_neighborhood(self.curr_sol)
            self.evaluate_neighborhood()
            self.sort_neighborhood()

            self.curr_sol = self.neighborhood[0]  # new solution becomes the best of the neighborhood
            self.search_history.append(copy.deepcopy(self.curr_sol))
            if prev_sol == self.curr_sol:
                print('Terminating after iteration number', it, ' because local extrema was found')
                return self.search_history, "local extrema", it

        print('Terminating because max iterations were exceeded, it = ', it)
        return self.search_history, "max iter exceeded", it

