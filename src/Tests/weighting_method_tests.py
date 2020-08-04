import numpy as np
from src.PreferenceArticulation.Solution import Solution
from src.PreferenceArticulation.BenchmarkObjectives import *
from src.TabuSearch.weighting_method import AposterioriWeightingMethod



def plot_BK1():



def example_1():
    params = dict(
        init_sol=Solution(np.array([9, 9])),
        problem=MOO_Problem.BK1,
        constraints=[MOO_Constraints.BK1_constraint],
        step_size=0.05,
        neighborhood_size=15,
        max_iter=2000,
        M=100,
        tabu_list_max_length=20,
        max_loops=100,
        search_space_dimensions=2,
        objective_space_dimensions=2,
        weights=[0.5, 0.5]
    )
    SearchInstance = AposterioriWeightingMethod(**params)
    result = SearchInstance.search()
    for i in result[1:]:
        print(i)



def main():
    example_1()


if __name__ == '__main__':
    main()

