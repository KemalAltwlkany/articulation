from LocalSearch.LS_apriori import LocalSearchApriori
from PreferenceArticulation.BenchmarkObjectives import MOO_Problem
from PreferenceArticulation.Solution import Solution
from PreferenceArticulation.Constraints import *
import numpy as np
import matplotlib.pyplot as plt
import random as random
import os as os
from datetime import datetime
import json as json

fig_num = 1


def save_test_to_file(file_name, init_sol, delta=None, max_iter=None, weights=None, M=None,
                      final_sol=None, seed_val=None, termination_reason=None, last_iter=None):
    data = {}
    # Create initial solution in json
    # Create x-vector in json
    data['Initial solution'] = {}
    data['Initial solution']['x'] = {}
    for ind, xi in enumerate(init_sol.x):
        s = 'x' + str(ind + 1)
        data['Initial solution']['x'][s] = xi

    # Create f-vector in json
    data['Initial solution']['f'] = {}
    for ind, fi in enumerate(init_sol.y):
        s = 'f' + str(ind + 1)
        data['Initial solution']['f'][s] = fi

    # Create other information
    data['delta'] = delta
    data['max iterations'] = max_iter
    data['M (criteria punishment)'] = M
    data['seed'] = seed_val
    data['termination reason'] = termination_reason
    data['last iter'] = last_iter

    # Create weights vector in json
    data['weights'] = {}
    for ind, wi in enumerate(weights):
        s = 'w' + str(ind + 1)
        data['weights'][s] = wi

    # Create final solution in json
    # Create x-vector in json
    data['Final solution'] = {}
    data['Final solution']['x'] = {}
    for ind, xi in enumerate(final_sol.x):
        s = 'x' + str(ind + 1)
        data['Final solution']['x'][s] = xi

    # Create f-vector in json
    data['Final solution']['f'] = {}
    for ind, fi in enumerate(final_sol.y):
        s = 'f' + str(ind + 1)
        data['Final solution']['f'][s] = fi

    with open(file_name + ".txt", 'w') as output:
        json.dump(data, output)

    datetime.today().strftime("%A, %d %B %Y") + " _BK1.txt"
    return


def LS_BK1_core(init_sol, delta, max_iter, constraints, weights, M, title, seed_val=0, save=False):

    # running Local Search on MOO Problem BK1
    problem = MOO_Problem.BK1
    search_alg = LocalSearchApriori(init_sol=init_sol, problem=problem, delta=delta, max_iter=max_iter,
                                    constraints=constraints, M=M, weights=weights, n_objectives=2)
    search_history, termination_reason, last_iter = search_alg.search()
    final_sol = search_history[-1]
    print('Final solution is: ', final_sol)

    # plotting the objective space
    global fig_num
    x1 = np.linspace(-5, 10, 300)
    x2 = np.linspace(-5, 10, 300)
    f1 = []
    f2 = []
    for i in x1:
        for j in x2:
            f1.append(i ** 2 + j ** 2)
            f2.append((i - 5) ** 2 + (j - 5) ** 2)
    plt.figure()
    plt.scatter(f1, f2, s=1.0)

    # second part plots the Pareto front of the problem
    x1 = np.linspace(0, 5, 50)
    x2 = np.linspace(0, 5, 50)
    f1 = x1 ** 2 + x2 ** 2
    f2 = (x1 - 5) ** 2 + (x2 - 5) ** 2
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='y')

    # plotting the search history
    f1 = []
    f2 = []
    for sol in search_history:
        f1.append(sol.y[0])
        f2.append(sol.y[1])
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='r')
    plt.plot([f1[0]], [f2[0]], marker=">", markersize=10, color='g')    # starting position (init sol)
    plt.plot([f1[-1]], [f2[-1]], marker="s", markersize=10, color='k')  # final position (final sol)

    # add some plot labels
    plt.title('Search history, Local Search, BK1 ' + title)
    plt.xlabel('f1(x1, x2)')
    plt.ylabel('f2(x1, x2)')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)

    # export data for reports
    if save is True:
        # SAVE PROCEDURE
        # 1.) Navigate to appropriate test folder
        # 2.) Check number of files. Every test generates 2 files, .txt and .png. The test ID is equal to n_files / 2
        # 3.) Save the plot
        # 4.) Save the test data in json format to txt file

        folder = "/home/kemal/Programming/Python/Preference_Articulation/Reports/TestResults/LocalSearch/LS_apriori/BK1"
        if not os.path.exists(folder):
            os.makedirs(folder)

        try:
            os.chdir(folder)
        except OSError:
            print('Could not cwd to: ', folder)
            print('Exiting.')
            return
        entries = os.listdir(folder)
        test_ID = len(entries) // 2
        # datetime.today().strftime("%A, %d %B %Y")
        file_name = "BK1_test_ID_" + str(test_ID)
        plt.savefig(file_name + '.png')
        save_test_to_file(file_name=file_name, init_sol=init_sol, delta=delta, max_iter=max_iter,
                          weights=weights, M=M, final_sol=final_sol, seed_val=seed_val,
                          termination_reason=termination_reason, last_iter=last_iter)
    # plt.show()
    fig_num = fig_num + 1


def LS_BK1_variations():
    seed_val = 1
    random.seed(seed_val)
    init_sol = Solution([random.uniform(-5, 10), random.uniform(-5, 10)])
    delta = 0.01
    max_iter = 3000
    constraints = [BoundaryConstraint([(-5, 10), (-5, 10)])]
    M = 100

    # A search with equal weights
    weights = [0.5, 0.5]
    LS_BK1_core(init_sol=init_sol, delta=delta, max_iter=max_iter, constraints=constraints, weights=weights, M=M,
                title='w=[0.5, 0.5]', seed_val=seed_val, save=True)

    # A search prioritizing the second criteria by a factor of 2
    weights = [0.33, 0.67]
    LS_BK1_core(init_sol=init_sol, delta=delta, max_iter=max_iter, constraints=constraints, weights=weights, M=M,
                title='w=[0.33, 0.67]', seed_val=seed_val, save=True)

    # A search prioritizing the first criteria by a factor of 2
    weights = [0.67, 0.33]
    LS_BK1_core(init_sol=init_sol, delta=delta, max_iter=max_iter, constraints=constraints, weights=weights, M=M,
                title='w=[0.67, 0.33]', seed_val=seed_val, save=True)

    # A search prioritizing the second criteria by a factor of 9
    weights = [0.1, 0.9]
    LS_BK1_core(init_sol=init_sol, delta=delta, max_iter=max_iter, constraints=constraints, weights=weights, M=M,
                title='w=[0.1, 0.9]', seed_val=seed_val, save=True)

    # A search prioritizing the first criteria by a factor of 9
    weights = [0.9, 0.1]
    LS_BK1_core(init_sol=init_sol, delta=delta, max_iter=max_iter, constraints=constraints, weights=weights, M=M,
                title='w=[0.9, 0.1]', seed_val=seed_val, save=True)


if __name__ == '__main__':
    LS_BK1_variations()
