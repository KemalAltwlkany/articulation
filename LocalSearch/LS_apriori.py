from LocalSearch.LS import LocalSearch


class LocalSearchApriori(LocalSearch):
    """
    Implements the apriori articulation of preferences type of LocalSearch. Apriori preferences are dealt with by
    simply aggregating all the objective functions into one objective function, optionally weighted. If no vector
    of weights is passed then it's assumed it's a vector of 1's, i.e. w[i] = 1, for every i.
    """

    n_objectives = 1
    weights = [1, 1]

    def __init__(self, init_sol=None, problem=None, delta=None, max_iter=None, constraints=None, M=100,
                 weights=None, n_objectives=None):
        super().__init__(init_sol=init_sol, problem=problem, delta=delta, max_iter=max_iter, constraints=constraints, M=M)
        self.weights = weights
        self.n_objectives = n_objectives
        if weights is None:
            self.weights = [1]*n_objectives  # in case no weights have been specified, use unity weights
        LocalSearchApriori.weights = self.weights
        LocalSearchApriori.n_objectives = n_objectives

    def sort_neighborhood(self):
        """In apriori articulation the set of objective functions is aggregated into one function, effectively making
        the multi-objective optimization problem (MOOP) a single-objective optimization problem (SOOP).
        Therefore, sorting the neighborhood is done by passing a key function to pythons built-in sort function.
        The key function simply performs the aggregation.
        """
        self.neighborhood.sort(key=LocalSearchApriori.compute_fitness)

    @staticmethod
    def compute_fitness(sol):
        """
        In apriori articulation, a full order can be introduced within the set of solutions being considered.
        The parameter/key which is used to establish full order is called the fitness of the solution.
        :param sol: 
        :return:
        """
        fit = 0
        for i in range(LocalSearchApriori.n_objectives):
            fit = fit + LocalSearchApriori.weights[i] * sol.y[i]
        return fit

