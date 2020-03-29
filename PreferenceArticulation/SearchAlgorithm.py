from PreferenceArticulation.ArticulationExceptions import ArticulationType, AbstractMethod
from PreferenceArticulation.Solution import Solution


class SearchAlgorithm:
    """Parent class for every Search Algorithm to be used.
    Since the focus shall be on continuous problems, with mathematically expressible objectives; every search algorithm
    contains the following attributes:
        *) init_sol - the initial solution, should be of type Solution.
        *) objectives - a list of objectives to be optimized. In case of single-objective optimization problems,
            provide the objectives parameter as a list containing one objective. Every objective should be a function
            which accepts a single argument - a list of floats.
        *) delta - the discretization step of the continuous space. Defines how and at which resolution/step size
            the adjacent solutions in the search space are generated.
        *) max_iter - the maximum allowed iterations before the algorithm terminates (unless an algorithm specific
            termination condition has been satisfied)
        *) constraints - constraints regarding the optimization problem. The treatment of constraints is algorithm
            specific.
    """

    def __init__(self, init_sol=None, problem=None, delta=None, max_iter=None, constraints=None, M=100):
        self.init_sol = init_sol    # class Solution
        self.problem = problem  # static method of class MOO_Problem
        self.delta = delta
        self.max_iter = max_iter
        self.constraints = constraints  # list of constraints
        self.neighborhood = []
        self.curr_sol = init_sol  # current best solution
        self.search_history = []    # list of all previous "current" solutions.
        self.M = M  # a large value, used to deteriorate the objectives in case of not solution not satisfying constraints

    # noinspection DuplicatedCode
    def generate_adjacent_x_vectors(self, x=[], neighborhood=[], current_span=[]):
        """
        Method returns a list of vectors which are in the first arguments neighborhood. The definition of neighborhood
        is assumed to be generating all possible permutations of the vector when varying each element of the vector
        by parameter self.delta (further noted as d).

        Example of neighborhood for vector [5]:
        [[5-d], [5], [5+d]]

        Example of neighborhood for vector [5, 7]:
        [ [5-d, 7-d], [5, 7-d], [5+d, 7-d], [5-d, 7], [5, 7], [5+d, 7], [5-d, 7+d], [5, 7+d], [5+d, 7+d] ]

        :param x:
        :param neighborhood: should never be specified by user
        :param current_span: should never be specified by user
        :return:
        """
        if len(x) == 1:
            span1 = list(current_span)
            span1.append(x[0] - self.delta)
            span2 = list(current_span)
            span2.append(x[0])
            span3 = list(current_span)
            span3.append(x[0] + self.delta)
            neighborhood.append(span1)
            neighborhood.append(span2)
            neighborhood.append(span3)
            return neighborhood
        else:
            new_span = list(current_span)
            new_span.append(x[0])
            neighborhood = self.generate_adjacent_x_vectors(x=x[1:], neighborhood=neighborhood, current_span=new_span)
            new_span[-1] = new_span[-1] - self.delta
            neighborhood = self.generate_adjacent_x_vectors(x=x[1:], neighborhood=neighborhood, current_span=new_span)
            new_span[-1] = new_span[-1] + 2 * self.delta
            neighborhood = self.generate_adjacent_x_vectors(x=x[1:], neighborhood=neighborhood, current_span=new_span)
        return neighborhood

    def generate_neighborhood(self, sol):
        # returns a list of vectors in SearchSpace which are in the neighborhood of solution sol.
        tmp = self.generate_adjacent_x_vectors(sol.x)
        # --IMPORTANT--
        # next line of code removes the original solution (sol) from the neighborhood!
        tmp.remove(sol.x)
        # generates objects of class Solution from x-vectors
        lst = []
        for x in tmp:
            lst.append(Solution(x))
        self.neighborhood = lst

    def evaluate_neighborhood(self):
        for sol in self.neighborhood:
            sol.y = self.evaluate_solution(sol)

    def evaluate_solution(self, sol):
        """
        Evaluating the solution means computing all the objective functions, i.e. filling the Solution.y vector
        :param sol: <class 'Solution'>
        :return: a list of floats, containing the evaluations of the objectives, defined by self.problem
        # self.problem is one of the static methods of class MOO_Problem which returns the specified list.
        """
        # no constraints, just return the vector of evaluated objective functions
        if self.constraints is None:
            return self.problem(sol.x)

        # if there are constraints, compute how many of them are not satisfied. Result is summed in punishment_val
        punishment_val = 0
        for constraint in self.constraints:
            satisfied, deteriorate_factor = constraint.is_satisfied(sol.x)
            if satisfied is True:
                continue
            else:
                punishment_val = punishment_val + deteriorate_factor

        # for every constraint not satisfied, deteriorate every objective by a factor of M
        punishment_val = punishment_val * self.M
        objectives = self.problem(sol.x)
        for i in range(len(objectives)):
            objectives[i] = objectives[i] + punishment_val
        return objectives

    # abstract
    def sort_neighborhood(self):
        raise AbstractMethod("Error! Method 'sort_neighborhood' is not overridden!")

    # abstract
    def search(self):
        raise AbstractMethod("Error! Method 'search' is not overridden!")

