import os as os
import numpy as np
import copy as copy
import pickle as pickle
from src.TabuSearch.TS import TabuSearch
from src.ProgressiveArticulation.rule_based_tree import train_RBTree


# In this implementation of progressive articulation a min-max single
# criterion function is minimized, i.e.:
# f = minimize max{f_{j}}
# where f_{j} is the value of the j-th criteria, j=1, 2,..., m
#
# Criterion f is modified in the sense that constraints are constantly added to the problem
# These constraints are provided by an IntelligentDM


class DynamicConstraints:
    # Every time the IDM decides uppon the constraints, these parameters get updated
    f1 = dict(
        A=0,
        B=0,
        C=0,
        D=0,
        E=0,
        F=0,
        previous_best=0
    )
    f2 = dict(
        A=0,
        B=0,
        C=0,
        D=0,
        E=0,
        F=0,
        previous_best=0
    )
    pen_val = 500

    @staticmethod
    def I1star_constraint(f):
        """
        This constraint favors improving f1 regardless of the deterioration of f2.
        :param f:
        :return:
        """
        if f[0] > DynamicConstraints.f1['D']:
            return DynamicConstraints.pen_val
        else:
            return 0

    @staticmethod
    def I2star_constraint(f):
        """
        This constraint favors improving f2 regardless of the deterioration of f1.
        :param f:
        :return:
        """
        if f[1] > DynamicConstraints.f2['D']:
            return DynamicConstraints.pen_val
        else:
            return 0

    @staticmethod
    def I1_constraint(f):
        """
        This method favors improving f1, provided that f2 remains at least as good.
        :param f:
        :return:
        """
        if f[0] > DynamicConstraints.f1['C'] or f[1] > DynamicConstraints.f2['previous_best']:
            return DynamicConstraints.pen_val
        else:
            return 0

    @staticmethod
    def I2_constraint(f):
        """
        This method favors improving f2, provided that f1 remains at least as good.
        :param f:
        :return:
        """
        if f[1] > DynamicConstraints.f2['C'] or f[0] > DynamicConstraints.f1['previous_best']:
            return DynamicConstraints.pen_val
        else:
            return 0

# The IntelligentDM expects two dicts of parameters -
# 1.) parameters regarding the search process (which is basically an analogy of the min-max goal programming
# 2.) parameters regarding the behaviour of the IntelligentDM which drives the search process multiple times.
# The IDM will run a search process after which it classifies the obtained results.
# Dependent on the classification, the IDM changes adds/removes constraints or modifies
# the search process parameters and runs it again.
# This is a simulation of progressive articulation.
class IntelligentDM:
    class_to_constraint_map = {
        'G': [],
        'I12': [],  # might need an upgrade on this decision here
        'I1': [DynamicConstraints.I1_constraint],
        'I2': [DynamicConstraints.I2_constraint],
        'I1*': [DynamicConstraints.I1star_constraint],
        'I2*': [DynamicConstraints.I2star_constraint]
    }

    def __init__(self, IDM_params=None, search_params=None):

        # Parameters regarding the search process
        self.search_params = copy.deepcopy(search_params)

        # IntelligentDM parameters
        self.problem_name = IDM_params['problem_name']
        self.n_repetitions = IDM_params['n_repetitions']
        self.pen_val = IDM_params['pen_val']
        self.total_runtime = 0
        self.tree = None
        self.f1 = None
        self.f2 = None
        self.dynamic_constraints = []

        # Parameters for post-analysis
        self.best_sols_history = []
        self.all_runtimes = []
        self.all_search_paths = []
        self.all_iter_numbers = []
        self.save = IDM_params['save']
        self.save_options = IDM_params['save_options']
        self.test_ID = copy.deepcopy(self.search_params['test_ID'])

        # Attributes used only for creating plots and txts (compatibility with previous articulation types)
        # for plots
        self.search_history = []
        self.global_best_sol = None
        self.termination_reason = None

        # for txts
        self.M = self.search_params['M']
        self.step_size = self.search_params['step_size']
        self.TL_max_length = self.search_params['tabu_list_max_length']
        self.neighborhood_size = self.search_params['neighborhood_size']
        self.max_iter = self.search_params['max_iter']
        self.max_loops = self.search_params['max_loops']
        self.seed_value = self.search_params['seed_value']  # this value will be used for every

        self.time_elapsed = None  # must manually sum all the runtimes
        self.init_sol = copy.deepcopy(self.search_params['init_sol'])
        self.curr_sol = None  # must manually make it equal to last item in last list of search_paths
        self.last_iter = 0
        self.total_reps = 0

    def save_results(self):
        # To make the results are compatible with previous articulation types
        # (in order to use the create plots and create txts functions):

        # combine all search results into one list
        for s in self.all_search_paths:
            self.search_history += s

        # update last solution found (in other articulation types, this is the last current sol)
        self.curr_sol = self.all_search_paths[-1][-1]

        # update total time elapsed by summing all elapsed times per each search
        self.time_elapsed = sum(self.all_runtimes)

        # update last iter by summing number of all iterations per each search
        self.last_iter = sum(self.all_iter_numbers)

        # Copied from TS
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

    def trainDM(self):
        # This method loads the articulate preferences/limits of individual criterias
        # and trains the decision tree.
        # f1 and f2 are return as dictionaries.
        # The key "previous_best" needs to be added manually (as the decision tree/csv contain no such info)
        self.tree, self.f1, self.f2 = train_RBTree(problem_name=self.problem_name, save=False, ret=True)
        self.f1['previous_best'] = 0
        self.f2['previous_best'] = 0

    def update_static_variables(self, f1_new_best=0, f2_new_best=0):
        # This method can later on be improved for more complex rules, such as reducing the limits
        # based on the number of iterations which had progressed.

        # Update new best solutions
        self.f1['previous_best'] = f1_new_best
        self.f2['previous_best'] = f2_new_best

        # Updates the static variables of class DynamicConstraints
        DynamicConstraints.f1 = copy.deepcopy(self.f1)
        DynamicConstraints.f2 = copy.deepcopy(self.f2)

    def update_constraints(self, classification):
        # Introduces new constraints (or removes existing)
        # based on how the RBDTree has classified the previous search result
        # In this particular implementation it is a very simple indexation from a static dictionary
        # but in general, it can be a more complex set of rules, hence why this function is created
        self.dynamic_constraints = IntelligentDM.class_to_constraint_map[classification]

    def procedure(self):
        DynamicConstraints.pen_val = self.pen_val
        self.trainDM()
        it = 0
        printing = True
        previous_sol_class = None
        class_of_curr_result = None
        while 1:
            self.search_params['dynamic_constraints'] = self.dynamic_constraints
            searchInstance = ProgressiveMinMaxProgramming(**self.search_params)
            results = searchInstance.search()

            # Memorize previous results for post analysis
            self.best_sols_history.append(copy.deepcopy(results['global_best_sol']))
            self.all_iter_numbers.append(results['last_iter'])
            self.all_runtimes.append(results['time_elapsed'])
            self.all_search_paths.append(copy.deepcopy(results['search_history']))

            # Based on results, IDM modifies parameters for the new search
            previous_sol_class = copy.deepcopy(class_of_curr_result)
            class_of_curr_result = self.tree.classify(results['global_best_sol'].get_y())

            self.update_constraints(class_of_curr_result)
            self.update_static_variables(f1_new_best=results['global_best_sol'].get_y()[0],
                                         f2_new_best=results['global_best_sol'].get_y()[1])

            # New initial solution becomes previous best solution
            self.search_params['init_sol'] = copy.deepcopy(results['global_best_sol'])

            it = it + 1

            # Update global best solution found
            if self.global_best_sol is None:
                self.global_best_sol = copy.deepcopy(results['global_best_sol'])
            elif self.global_best_sol.get_val() > results['global_best_sol'].get_val():
                self.global_best_sol = copy.deepcopy(results['global_best_sol'])
            elif class_of_curr_result == 'G' and previous_sol_class == 'G':
                self.termination_reason = 'No improvement while in class G for two iters.'
                break

            if it > self.n_repetitions:
                self.termination_reason = 'Maximum iters exceeded.'
                break
            # FEATURE - add a termination reason if two successive solutions are classified as 'G'
            # perhaps a measure of improvement in terms of 5% difference?
            # that could be after elif, as else:... because that implies the new found solution isn't
            # better than the current one. So if the current one is from class G - terminate.

            if printing is True:
                print('--------------------------------------------------------------------------------')
                print('Finished repetition: ', it)
                print('Best solution found so far: ', results['global_best_sol'])
                print('Solution has been classified as: ', class_of_curr_result)
                print('The solution had been found in: ', results['last_iter'], ' iterations')
                print('The termination reason was: ', results['termination_reason'])

        print('Termination reason of IntelligentDM: ', self.termination_reason)
        self.total_reps = it
        if self.save is True:
            self.save_results()



class ProgressiveMinMaxProgramming(TabuSearch):

    def __init__(self, init_sol=None, problem=None, constraints=None, step_size=None, neighborhood_size=None,
                 max_iter=None, M=None, tabu_list_max_length=None, max_loops=None, search_space_dimensions=None,
                 objective_space_dimensions=None, save=False, save_options=None, seed_value=0, test_ID=None,
                 dynamic_constraints=None):
        super().__init__(init_sol, problem, constraints, step_size, neighborhood_size, max_iter, M,
                         tabu_list_max_length, max_loops, search_space_dimensions, objective_space_dimensions, save,
                         save_options, seed_value, test_ID)

        self.dynamic_constraints = list(dynamic_constraints)  # list of functions which return penalty values.

    def evaluate_solution(self, sol):
        """
        In this approach of progressive min-max progrmming, the largest criterion value is
        minimized, i.e.:
        f = minimize max {f_{j}}, where j=1, 2,..., m
        Rigid constraints remain the same as in previous implementations, with the exception of
        the possible existence of new - so called dynamic constraints.
        These are introduced by the IntelligentDM.
        :param sol:
        :return:
        """
        sol.set_val(np.max(sol.get_y()) + self.penalty(sol) + self.dynamic_penatly(sol))

    def dynamic_penatly(self, sol):
        penalty = 0
        for dynConstr in self.dynamic_constraints:
            penalty += dynConstr(sol.get_y())
        return penalty

