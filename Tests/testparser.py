# parsing necessities
import sys as sys
import getopt as getopt

# working with files
import os as os
import json as json
import copy as copy
from datetime import datetime

# packages needed for the algorithm
import numpy as np
import matplotlib.pyplot as plt
import random as random
import math as math

# my modules
sys.path.insert(0, "/home/kemal/Programming/Python/Articulation")
from PreferenceArticulation.Solution import Solution
from PreferenceArticulation.BenchmarkObjectives import MOO_Problem
from PreferenceArticulation.Constraints import BoundaryConstraint
from TabuSearch.TS_apriori import TabuSearchApriori
from LocalSearch.LS_apriori import LocalSearchApriori


fig_num = 1

def save_test_to_file(init_sol, delta, max_iter, M, tabu_list_max_length, weights, max_loops, min_progress, final_sol,
                      seed_val, termination_reason, last_iter, file_name):
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
    data['Seed'] = seed_val
    data['Termination reason'] = termination_reason
    data['Last iter'] = last_iter
    data['TabuList max length'] = tabu_list_max_length
    data['max loops'] = max_loops
    data['min progress'] = min_progress

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

    return


# For every test, if the parameter Manual is True, then the test settings can be found in the directory of that
# algorithm+benchmark problem. i.e.;
# /Articulation/Tests/standardized_tests/TS/BK/[testSettingsFile.txt]
# All testSettingsFiles are named standard_i.txt, where i denotes the unique identifier of that test.
# i is specified by parameter mantype, passed to every function as well.


# NOTE - I can modify these functions to add whether I want to save them to an Excel spreadhseet, or if I even want to
# plot the results, etc...
def TS_BK1_core(manual=False, std_ID=None, seed_val=0, toPlot=True, save=False):
    init_sol = delta = max_iter = M = tabu_list_max_length = weights = max_loops = min_progress = None

    # Setting up the algorithm parameters
    if manual is True:
        pass
        # need to implement this
    else:
        # Boundaries for problem BK1 are x1 € [-5, 10], x2 € [-5, 10]
        random.seed(seed_val)
        init_sol = Solution([random.uniform(-5, 10), random.uniform(-5, 10)])
        delta = 0.01
        max_iter = 500
        max_loops = 15
        min_progress = 0.001
        tabu_list_max_length = 20
        M = 100
        # random weights
        a = round(random.uniform(0, 1), 2)
        weights = [a, round(1 - a, 2)]

    problem = MOO_Problem.BK1
    constraints = [BoundaryConstraint([(-5, 10), (-5, 10)])]
    search_alg = TabuSearchApriori(init_sol=init_sol, problem=problem, delta=delta, max_iter=max_iter,
                                   constraints=constraints, M=M, tabu_list_max_length=tabu_list_max_length,
                                   weights=weights, n_objectives=2, max_loops=max_loops, min_progress=min_progress)

    # running Tabu Search on MOO Problem BK1
    search_history, termination_reason, last_iter = search_alg.search()
    final_sol = search_history[-1]
    print('Final solution is: ', final_sol)

    # plotting the objective space and the results (search history, etc...)
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
    plt.plot([f1[0]], [f2[0]], marker=">", markersize=10, color='g')  # starting position (init sol)
    plt.plot([f1[-1]], [f2[-1]], marker="s", markersize=10, color='k')  # final position (final sol)

    # add some plot labels
    # To every title I append the weights information as well, so firstly lets convert that to a string
    w_str = 'w=[' + str(weights[0]) + ", " + str(weights[1]) + "]"
    plt.title('Search history, Tabu Search, BK1 ' + w_str)
    plt.xlabel('f1(x1, x2)')
    plt.ylabel('f2(x1, x2)')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.grid(True)

    # export data to json/txt in order to generate reports
    if save is True:
        # SAVE PROCEDURE
        # 1.) Navigate to appropriate test folder
        # 2.) Check number of files. Every test generates 2 files, .txt and .png. The test ID is equal to n_files / 2
        # 3.) Save the plot
        # 4.) Save the test data in json format to txt file

        folder = "/home/kemal/Programming/Python/Articulation/Tests/test_results_raw/TS/TS_apriori/BK1"
        if not os.path.exists(folder):
            os.makedirs(folder)

        try:
            os.chdir(folder)
        except OSError:
            print('Could not cwd to: ', folder)
            print('Exiting with status flag 83.')
            sys.exit(83)

        entries = os.listdir(folder)
        test_ID = len(entries) // 2
        file_name = "BK1_test_ID_" + str(test_ID)
        plt.savefig(file_name + '.png')
        save_test_to_file(init_sol=init_sol, delta=delta, max_iter=max_iter, M=M, tabu_list_max_length=tabu_list_max_length,
                          weights=weights, max_loops=max_loops, min_progress=min_progress, final_sol=final_sol, seed_val=seed_val,
                          termination_reason=termination_reason, last_iter=last_iter, file_name=file_name)
    # plt.show()
    fig_num = fig_num + 1


def FON():
    print('Executed test FON')


def BK1():
    print('Executed test BK1')


def SCH1():
    print('Executed test SCH1')


def main():
    args = copy.deepcopy(sys.argv[1:])  # the 0-th argument is the name of the file passed as argument
    long_options = ['manual=', 'seed=', 'problem=', 'stdID=', 'save=']
    try:
        optlist, args2 = getopt.getopt(args, '', long_options)
    except getopt.GetoptError as err:
        print('Error while parsing the arguments! Details: ')
        print(str(err))
        print('Exiting program, with value 5.')
        sys.exit(5)
    print(args)
    print(optlist)
    # convert the optlist from list to dict
    data = {}
    for opt, val in optlist:
        data[opt] = val

    # interpret the parsed options
    seed_val_ = int(data['--seed'])
    std_ID_ = data['--stdID']
    prob = data['--problem']
    man = None
    if data['--manual'] == 'T':
        man = True
    else:
        man = False

    if data['--save'] == 'T':
        save = True
    else:
        save = False

    if prob == 'FON':
        FON()
    elif prob == 'BK1':
        TS_BK1_core(manual=man, std_ID=std_ID_, seed_val=seed_val_, toPlot=True, save=save)
    elif prob == 'SCH1':
        SCH1()
    else:
        print('Error! Did not find benchmark problem name! Exiting with value 10.')
        sys.exit(10)


if __name__ == '__main__':
    main()
