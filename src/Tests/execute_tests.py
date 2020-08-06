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

    parser.add_argument("--seed", default=0, help="Random seed value.", type=int)
    parser.add_argument("--step_size", default=0.5, help="Step size.", type=float)
    parser.add_argument("--neighborhood_size", default=20, help="Neighborhood size.", type=int)
    parser.add_argument("--TL_len", default=25, help="Tabu list length", type=int)
    parser.add_argument("--max_iter", default=1000, help="Maximum iterations.", type=int)
    parser.add_argument("--max_loops", default=150, help="Maximum loops with no progress.", type=int)
    parser.add_argument("--M", default=100, help="Penalty value for violated constraints.", type=int)
    parser.add_argument("--nObj", default=2, help="Number of objectives-", type=int)
    parser.add_argument("--nAlt", default=2, help="Number of search space alternatives.", type=int)
    parser.add_argument("--save", default=True, help="Specifies whether test results should be saved.", type=bool)


    # Specific and longer parameters
    parser.add_argument("--load_params", default=False, help="Defines whether to load algorithm parameters.", type=bool)

    

    #parser.add_argument("preference_type", help='Type of preference articulation used', type=str)
    args = parser.parse_args()
    print(args)


if __name__ == '__main__':
    main()


