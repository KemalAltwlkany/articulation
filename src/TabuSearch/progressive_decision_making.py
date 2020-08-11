import numpy as np
import copy as copy
from src.TabuSearch.TS import TabuSearch

# Code is not complete.


# Code can be optimized A LOT
# i.e.:
# -no need to pass all zones, only B3 and B2 (which if passed as arrays can be used with numpy for vectorization)

class ProgressiveArticulationDecisionMaker:

    def __init__(self, DM_params=None, search_params=None):
        # Settings for one min-max search
        self.search_params = copy.deepcopy(search_params)

        # Setup search parameters
        self.max_repetitions = DM_params['max_repetitions']
        self.unacceptable_penalty = DM_params['unacceptable_penalty']
        self.unsatisficing_penalty = DM_params['unsatisficing_penalty']
        self.step = DM_params['narrow_step']

        self.curr_centroids = copy.deepcopy(DM_params['centroids'])
        self.centroids_history = []
        self.m = len(self.curr_centroids)  # Extract number of objectives from length of aspirations list

        self.aspirations = [None] * self.m
        self.aspiration_history = []

        self.curr_zones = [None] * self.m
        self.zones_history = []

        self.all_searches = []

    def update_centroids(self, it):
        # Every iteration, "d" is decreased and "a" is increased in order to narrow down the neutral zone

        # WARNING - ensure b =< d always and b >= a
        for i in range(self.m):
            if self.curr_centroids[i]['d'] < self.curr_centroids[i]['b']:
                self.curr_centroids[i]['d'] = self.curr_centroids[i]['b']
            else:
                self.curr_centroids[i]['d'] = self.curr_centroids[i]['d'] - it * self.step
            if self.curr_centroids[i]['a'] > self.curr_centroids[i]['b']:
                self.curr_centroids[i]['a'] = self.curr_centroids[i]['b']
            else:
                self.curr_centroids[i]['a'] = self.curr_centroids[i]['a'] + it * self.step

        # For post-analysis, memorize all the aspiration levels.
        self.centroids_history.append(copy.deepcopy(self.curr_centroids))

    def update_zones(self):
        for i in range(self.m):
            B3 = (self.curr_centroids[i]['e'] + self.curr_centroids[i]['d']) / 2
            B2 = (self.curr_centroids[i]['d'] + self.curr_centroids[i]['c']) / 2
            B1 = (self.curr_centroids[i]['c'] + self.curr_centroids[i]['b']) / 2
            B0 = (self.curr_centroids[i]['b'] + self.curr_centroids[i]['a']) / 2
            A = self.curr_centroids[i]['b']
            self.curr_zones[i] = dict(
                A=A,
                B0=B0,
                B1=B1,
                B2=B2,
                B3=B3
            )
        # For post-analysis, memorize all the zones
        self.zones_history.append(copy.deepcopy(self.curr_zones))

    def update_aspirations(self):
        for i in range(self.m):
            self.aspirations[i] = self.curr_centroids[i]['a']
        # For post-analysis, memorize all the aspirations
        self.aspiration_history.append(copy.deepcopy(self.aspirations))

    def search(self):
        # Initialize zones and aspirations (centroids are initialized within constructor)
        self.update_zones()
        self.update_aspirations()
        self.centroids_history.append(copy.deepcopy(self.curr_centroids))  # manually save the initial centroids
        it = 0

        # Repeat the search self.max_rep - times.
        progressive_params = dict()
        progressive_params['unacceptable_penalty'] = self.unacceptable_penalty
        progressive_params['unsatisficing_penalty'] = self.unsatisficing_penalty
        while it < self.max_repetitions:
            progressive_params['zones'] = self.curr_zones
            progressive_params['aspirations'] = self.aspirations
            self.search_params['progressive_params'] = progressive_params
            print('************************* REPETITION NUMBER ', it, ' *********************************')
            searchInstance = ProgressiveArticulationProgramming(**self.search_params)
            return_dict = searchInstance.search()

            print('Current aspiration levels: ', self.aspirations)
            print('Current zones: ', self.curr_zones)
            last_best_sol_found = return_dict['global_best_sol']
            print('Best solution found: ')
            print(last_best_sol_found)
            print('*********************************************************************')
            if self.verify_termination_conditions(last_best_sol_found):
                print('Aspiration zone reached.')
                return
            it = it + 1
            self.update_centroids(it)
            self.update_zones()
            self.update_aspirations()
        print('Maximum repetitions exceeded!')

    def verify_termination_conditions(self, sol):
        # Verify whether last best solution is better than all aspiration levels
        y = sol.get_y()
        for i in range(self.m):
            if y[i] > self.aspirations[i]:
                return False
        return True



# The decision making process is a combination of a priori and progressive articulation of preferences
class ProgressiveArticulationProgramming(TabuSearch):

    def __init__(self, init_sol=None, problem=None, constraints=None, step_size=None, neighborhood_size=None,
                 max_iter=None, M=None, tabu_list_max_length=None, max_loops=None, search_space_dimensions=None,
                 objective_space_dimensions=None, save=False, save_options=None, seed_value=0, test_ID=None, progressive_params=None):
        super().__init__(init_sol, problem, constraints, step_size, neighborhood_size, max_iter, M,
                         tabu_list_max_length, max_loops, search_space_dimensions, objective_space_dimensions, save,
                         save_options, seed_value, test_ID)

        self.aspirations = np.array(progressive_params['aspirations'])
        self.zones = copy.deepcopy(progressive_params['zones'])
        self.unacceptable_penalty = progressive_params['unacceptable_penalty']
        self.unsatisficing_penalty = progressive_params['unsatisficing_penalty']



    def zone_penalty(self, y):
        # For every criteria, determine in which zone it lies and compute the penalty accordingly
        penalty = 0
        for i in range(self.m):  # m = number of criteria
            if y[i] > self.zones[i]['B3']:
                penalty += self.unacceptable_penalty
            elif y[i] > self.zones[i]['B2']:
                penalty += self.unsatisficing_penalty
        return penalty

    def evaluate_solution(self, sol):
        sol.set_val(np.max(np.subtract(sol.get_y(), self.aspirations)) + self.penalty(sol) + self.zone_penalty(sol.get_y()))









