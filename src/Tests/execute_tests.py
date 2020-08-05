# parsing necessities
# import sys as sys
# import getopt as getopt

# packages
# import numpy as np
# import copy as copy

# my modules
# sys.path.insert(0, "/home/kemal/Programming/Python/Articulation")
# from src.PreferenceArticulation.Solution import Solution
# from src.PreferenceArticulation.BenchmarkObjectives import *
# from src.TabuSearch.weighting_method import AposterioriWeightingMethod


# def BK1_aposteriori(additional_options):
#     #additional_options
#     params = dict(
#         init_sol=Solution(np.array([9, 9])),
#         problem=MOO_Problem.BK1,
#         constraints=[MOO_Constraints.BK1_constraint],
#         step_size=0.05,
#         neighborhood_size=15,
#         max_iter=2000,
#         M=100,
#         tabu_list_max_length=20,
#         max_loops=100,
#         search_space_dimensions=2,
#         objective_space_dimensions=2,
#         weights=[0.5, 0.5]
#     )
import argparse as argparse

def main():
    parser = argparse.ArgumentParser(description='This module/script is used to run a single test.')

    # group1 - used for declaring the type of articulation preference
    group1 = parser.add_mutually_exclusive_group()
    group1.add_argument("--apriori", help='A priori articulation of preferences used.', action='store_true')
    group1.add_argument("--aposteriori", help='A posteriori articulation of preferences used.', action='store_true')
    group1.add_argument("--progressive", help='Progressive articulation of preferences used', action='store_true')

    # group2 - used for declaring the test problem used
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument("--BK1", help='Test problem BK1.', action='store_true')
    group2.add_argument("--IM1", help='Test problem IM1.', action='store_true')
    group2.add_argument("--FON", help='Test problem FON.', action='store_true')
    group2.add_argument("--SCH1", help='Test problem SCH1.', action='store_true')

    

    #parser.add_argument("preference_type", help='Type of preference articulation used', type=str)
    args = parser.parse_args()
    print(args)


if __name__ == '__main__':
    main()


