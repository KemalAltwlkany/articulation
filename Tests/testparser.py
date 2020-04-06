# parsing necessities
import sys as sys
import getopt as getopt

# working with files
import os as os
import copy as copy
import json as json

# packages needed for the algorithm
import numpy as np
import matplotlib.pyplot as plt
import random as random
import math as math

# my modules
sys.path.insert(0, "/home/kemal/Programming/Python/Articulation")
from PreferenceArticulation.Solution import Solution
from PreferenceArticulation.BenchmarkObjectives import MOO_Problem
from PreferenceArticulation.Constraints import *
from TabuSearch.TS_apriori import TabuSearchApriori
from LocalSearch.LS_apriori import LocalSearchApriori


fig_num = 1

def save_test_to_file_vol(init_sol, delta, max_iter, M, tabu_list_max_length, weights, max_loops, min_progress, final_sol,
                          seed_val, termination_reason, last_iter, file_name):
    f = open(file_name + ".txt", "w")
    # fpdf library ignores \t and \n's, so we embedded indentation by adding it manually
    wh = "        "  # an eight-character width whitespace (2xtab)
    f.write('----------------------------------------------Algorithm parameters----------------------------------------------'  + "\n")
    f.write(wh + 'Delta = == ' + str(delta) + "\n")
    f.write(wh + 'Max iterations = == ' + str(max_iter) + "\n")
    f.write(wh + 'M (criteria punishment) = == ' + str(M) + "\n")
    f.write(wh + 'Seed = == ' + str(seed_val) + "\n")
    f.write(wh + 'Max loops = == ' + str(max_loops) + "\n")
    f.write(wh + 'Min progress = == ' + str(min_progress) + "\n")
    f.write(wh + 'Tabu list max length = == ' + str(tabu_list_max_length) + "\n")
    f.write(wh + 'Weights = {' + "\n")
    for ind, wi in enumerate(weights):
        f.write(wh + wh + 'w' + str(ind + 1) + " = == " + str(wi) + "\n")
    f.write(wh + "}\n")
    f.write("----------------------------------------------Performance----------------------------------------------------------\n")
    f.write(wh + 'Termination reason = == ' + str(termination_reason) + "\n")
    f.write(wh + "Last iteration = == " + str(last_iter) + "\n")
    f.write("---------------------------------------------------------------------\n")
    f.write(wh + 'Initial solution:' + "\n")
    f.write(wh + wh + 'x = { ' + "\n")
    for ind, xi in enumerate(init_sol.x):
        f.write(wh + wh + wh + 'x' + str(ind + 1) + " = == " + str(xi) + "\n")
    f.write(wh + wh + '}\n')
    f.write(wh + wh + 'f = {\n')
    for ind, fi in enumerate(init_sol.y):
        f.write(wh + wh + wh + 'f' + str(ind + 1) + " = == " + str(fi) + "\n")
    f.write(wh + wh + '}\n')
    f.write("---------------------------------------------------------------------\n")
    f.write(wh + 'Final solution:\n')
    f.write(wh + wh + 'x = {\n')
    for ind, xi in enumerate(final_sol.x):
        f.write(wh + wh + wh + 'x' + str(ind + 1) + " = == " + str(xi) + "\n")
    f.write(wh + wh + '}\n')
    f.write(wh + wh + 'f = {\n')
    for ind, fi in enumerate(final_sol.y):
        f.write(wh + wh + wh + 'f' + str(ind + 1) + " = == " + str(fi) + "\n")
    f.write(wh + wh + '}')
    f.close()



# For every test, if the parameter Manual is True, then the test settings can be found in the directory of that
# algorithm+benchmark problem. i.e.;
# /Articulation/Tests/standardized_tests/TS/BK1/[testSettingsFile.txt]
# All testSettingsFiles are named standard_i.txt, where i denotes the unique identifier of that test.
# i is specified by parameter mantype, passed to every function as well.


# Given the file_name (complete, with path), loads the standardized test parameters from the txt file written in json format
# and dumps them to dictionary. Returns all values of dictionary.
def load_standardized_test(f_name):
    data = {}
    with open(f_name, 'r') as json_file:
        data = json.load(json_file)
    return data["init_sol"], data["delta"], data["max_iter"], data["M"], data["tabu_list_max_length"], data["weights"], data["max_loops"], data["min_progress"], data["description"]


# NOTE - I can modify these functions to add whether I want to save them to an Excel spreadhseet, or to confirm whether
# I even want to plot the results, etc...
def TS_BK1_core(manual=False, std_ID=None, seed_val=0, toPlot=True, save=False):
    init_sol = delta = max_iter = M = tabu_list_max_length = weights = max_loops = min_progress = description = None
    # description not used for now.
    # Setting up the algorithm parameters
    if manual is True:
        # read the standard test setup from file
        # /Articulation/Tests/standardized_tests/TS/BK1/standard_i.txt, where i is = std_ID
        # load the json formatted data into a dictionary
        # Get the data from the the function load_standardized_test
        f_name = "/home/kemal/Programming/Python/Articulation/Tests/standardized_tests/TS/BK1/standard_" + str(std_ID) + ".txt"
        init_sol_x_vect, delta, max_iter, M, tabu_list_max_length, weights, max_loops, min_progress, description = load_standardized_test(f_name)
        init_sol = Solution(init_sol_x_vect)
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
    search_history, termination_reason, last_iter, glob_sol = search_alg.search()
    final_sol = glob_sol
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
    plt.scatter(f1, f2, s=1.0, color='dodgerblue')

    # second part plots the Pareto front of the problem
    x1 = np.linspace(0, 5, 50)
    x2 = np.linspace(0, 5, 50)
    f1 = x1 ** 2 + x2 ** 2
    f2 = (x1 - 5) ** 2 + (x2 - 5) ** 2
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='navy')

    # plotting the search history
    f1 = []
    f2 = []
    for sol in search_history:
        f1.append(sol.y[0])
        f2.append(sol.y[1])
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='red')
    plt.plot([f1[0]], [f2[0]], marker=">", markersize=15, color='darkgreen')  # starting position (init sol)
    plt.plot([f1[-1]], [f2[-1]], marker="s", markersize=15, color='darkgreen')  # final position (final sol)
    plt.plot(final_sol.y[0], final_sol.y[1], marker="*", markersize=15, color='gold')
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
        save_test_to_file_vol(init_sol=init_sol, delta=delta, max_iter=max_iter, M=M, tabu_list_max_length=tabu_list_max_length,
                          weights=weights, max_loops=max_loops, min_progress=min_progress, final_sol=final_sol, seed_val=seed_val,
                          termination_reason=termination_reason, last_iter=last_iter, file_name=file_name)
    # plt.show()
    fig_num = fig_num + 1


def TS_IM1_core(manual=False, std_ID=None, seed_val=0, toPlot=True, save=False):
    init_sol = delta = max_iter = M = tabu_list_max_length = weights = max_loops = min_progress = description = None
    # description not used for now.
    # Setting up the algorithm parameters
    if manual is True:
        # read the standard test setup from file
        # /Articulation/Tests/standardized_tests/TS/IM1/standard_i.txt, where i is = std_ID
        # load the json formatted data into a dictionary
        # Get the data from the the function load_standardized_test
        f_name = "/home/kemal/Programming/Python/Articulation/Tests/standardized_tests/TS/IM1/standard_" + str(std_ID) + ".txt"
        init_sol_x_vect, delta, max_iter, M, tabu_list_max_length, weights, max_loops, min_progress, description = load_standardized_test(f_name)
        init_sol = Solution(init_sol_x_vect)
    else:
        # Boundaries for problem IM1 are x1 € [1, 4], x2 € [1, 2]
        random.seed(seed_val)
        init_sol = Solution([random.uniform(1, 4), random.uniform(1, 2)])
        delta = 0.001
        max_iter = 500
        max_loops = 15
        min_progress = 0.0001
        tabu_list_max_length = 20
        M = 100
        # random weights
        a = round(random.uniform(0, 1), 2)
        weights = [a, round(1 - a, 2)]

    problem = MOO_Problem.IM1
    constraints = [BoundaryConstraint([(1, 4), (1, 2)])]
    search_alg = TabuSearchApriori(init_sol=init_sol, problem=problem, delta=delta, max_iter=max_iter,
                                   constraints=constraints, M=M, tabu_list_max_length=tabu_list_max_length,
                                   weights=weights, n_objectives=2, max_loops=max_loops, min_progress=min_progress)

    # running Tabu Search on MOO Problem IM1
    search_history, termination_reason, last_iter, glob_sol = search_alg.search()
    final_sol = glob_sol
    print('Final solution is: ', final_sol)

    # plotting the objective space and the results (search history, etc...)
    global fig_num
    x1 = np.linspace(1, 4, 300)
    x2 = np.linspace(1, 2, 300)
    f1 = []
    f2 = []
    for i in x1:
        for j in x2:
            f1.append(2 * math.sqrt(i))
            f2.append(i * (1 - j) + 5)
    plt.figure()
    plt.scatter(f1, f2, s=1.0, color='dodgerblue')

    # second part plots the Pareto front of the problem
    f1 = 2*np.sqrt(x1)
    x2 = 2
    f2 = x1 * (1 - x2) + 5
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='navy')

    # plotting the search history
    f1 = []
    f2 = []
    for sol in search_history:
        f1.append(sol.y[0])
        f2.append(sol.y[1])
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='red')
    plt.plot([f1[0]], [f2[0]], marker=">", markersize=15, color='darkgreen')  # starting position (init sol)
    plt.plot([f1[-1]], [f2[-1]], marker="s", markersize=15, color='darkgreen')  # final position (final sol)
    plt.plot(final_sol.y[0], final_sol.y[1], marker="*", markersize=15, color='gold')

    # add some plot labels
    # To every title I append the weights information as well, so firstly lets convert that to a string
    w_str = 'w=[' + str(weights[0]) + ", " + str(weights[1]) + "]"
    plt.title('Search history, Tabu Search, IM1 ' + w_str)
    plt.xlabel('f1(x1, x2)')
    plt.ylabel('f2(x1, x2)')
    plt.xlim(0, 5)
    plt.ylim(0, 5)
    plt.grid(True)

    # export data to json/txt in order to generate reports
    if save is True:
        # SAVE PROCEDURE
        # 1.) Navigate to appropriate test folder
        # 2.) Check number of files. Every test generates 2 files, .txt and .png. The test ID is equal to n_files / 2
        # 3.) Save the plot
        # 4.) Save the test data in json format to txt file

        folder = "/home/kemal/Programming/Python/Articulation/Tests/test_results_raw/TS/TS_apriori/IM1"
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
        file_name = "IM1_test_ID_" + str(test_ID)
        plt.savefig(file_name + '.png')
        save_test_to_file_vol(init_sol=init_sol, delta=delta, max_iter=max_iter, M=M, tabu_list_max_length=tabu_list_max_length,
                          weights=weights, max_loops=max_loops, min_progress=min_progress, final_sol=final_sol, seed_val=seed_val,
                          termination_reason=termination_reason, last_iter=last_iter, file_name=file_name)
    # plt.show()
    fig_num = fig_num + 1


def TS_SCH1_core(manual=False, std_ID=None, seed_val=0, toPlot=True, save=False):
    init_sol = delta = max_iter = M = tabu_list_max_length = weights = max_loops = min_progress = description = None
    # description not used for now.
    # Setting up the algorithm parameters
    if manual is True:
        # read the standard test setup from file
        # /Articulation/Tests/standardized_tests/TS/SCH1/standard_i.txt, where i is = std_ID
        # load the json formatted data into a dictionary
        # Get the data from the the function load_standardized_test
        f_name = "/home/kemal/Programming/Python/Articulation/Tests/standardized_tests/TS/SCH1/standard_" + str(std_ID) + ".txt"
        init_sol_x_vect, delta, max_iter, M, tabu_list_max_length, weights, max_loops, min_progress, description = load_standardized_test(f_name)
        init_sol = Solution(init_sol_x_vect)
    else:
        # Boundaries for problem SCH1 are x1 € [-10, 10]
        random.seed(seed_val)
        init_sol = Solution([random.uniform(-10, 10)])
        delta = 0.01
        max_iter = 500
        max_loops = 15
        min_progress = 0.001
        tabu_list_max_length = 20
        M = 100
        # random weights
        a = round(random.uniform(0, 1), 2)
        weights = [a, round(1 - a, 2)]

    problem = MOO_Problem.SCH1
    constraints = [BoundaryConstraint([(-10, 10)])]
    search_alg = TabuSearchApriori(init_sol=init_sol, problem=problem, delta=delta, max_iter=max_iter,
                                   constraints=constraints, M=M, tabu_list_max_length=tabu_list_max_length,
                                   weights=weights, n_objectives=2, max_loops=max_loops, min_progress=min_progress)

    # running Tabu Search on MOO Problem SCH1
    search_history, termination_reason, last_iter, glob_sol = search_alg.search()
    final_sol = glob_sol
    print('Final solution is: ', final_sol)

    # plotting the objective space and the results (search history, etc...)
    global fig_num
    x1 = np.linspace(-10, 10, 1000)
    f1 = []
    f2 = []
    for i in x1:
        f1.append(i ** 2)
        f2.append((i - 2) ** 2)
    plt.figure()
    plt.scatter(f1, f2, s=1.0, color='dodgerblue')

    # second part plots the Pareto front of the problem
    x1 = np.linspace(0, 2, 80)
    f1 = x1**2
    f2 = (x1 - 2)**2
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='navy')

    # plotting the search history
    f1 = []
    f2 = []
    for sol in search_history:
        f1.append(sol.y[0])
        f2.append(sol.y[1])
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='red')
    plt.plot([f1[0]], [f2[0]], marker=">", markersize=15, color='darkgreen')  # starting position (init sol)
    plt.plot([f1[-1]], [f2[-1]], marker="s", markersize=15, color='darkgreen')  # final position (final sol)
    plt.plot(final_sol.y[0], final_sol.y[1], marker="*", markersize=15, color='gold')

    # add some plot labels
    # To every title I append the weights information as well, so firstly lets convert that to a string
    w_str = 'w=[' + str(weights[0]) + ", " + str(weights[1]) + "]"
    plt.title('Search history, Tabu Search, SCH1 ' + w_str)
    plt.xlabel('f1(x1, x2)')
    plt.ylabel('f2(x1, x2)')
    plt.xlim(0, 60)
    plt.ylim(0, 60)
    plt.grid(True)

    # export data to json/txt in order to generate reports
    if save is True:
        # SAVE PROCEDURE
        # 1.) Navigate to appropriate test folder
        # 2.) Check number of files. Every test generates 2 files, .txt and .png. The test ID is equal to n_files / 2
        # 3.) Save the plot
        # 4.) Save the test data in json format to txt file

        folder = "/home/kemal/Programming/Python/Articulation/Tests/test_results_raw/TS/TS_apriori/SCH1"
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
        file_name = "SCH1_test_ID_" + str(test_ID)
        plt.savefig(file_name + '.png')
        save_test_to_file_vol(init_sol=init_sol, delta=delta, max_iter=max_iter, M=M, tabu_list_max_length=tabu_list_max_length,
                          weights=weights, max_loops=max_loops, min_progress=min_progress, final_sol=final_sol, seed_val=seed_val,
                          termination_reason=termination_reason, last_iter=last_iter, file_name=file_name)
    # plt.show()
    fig_num = fig_num + 1


def TS_FON_core(manual=False, std_ID=None, seed_val=0, toPlot=True, save=False):
    init_sol = delta = max_iter = M = tabu_list_max_length = weights = max_loops = min_progress = description = None
    # description not used for now.
    # Setting up the algorithm parameters
    if manual is True:
        # read the standard test setup from file
        # /Articulation/Tests/standardized_tests/TS/SCH1/standard_i.txt, where i is = std_ID
        # load the json formatted data into a dictionary
        # Get the data from the the function load_standardized_test
        f_name = "/home/kemal/Programming/Python/Articulation/Tests/standardized_tests/TS/FON/standard_" + str(std_ID) + ".txt"
        init_sol_x_vect, delta, max_iter, M, tabu_list_max_length, weights, max_loops, min_progress, description = load_standardized_test(f_name)
        init_sol = Solution(init_sol_x_vect)
    else:
        # Boundaries for problem FON are xi € [-4, 4]
        random.seed(seed_val)
        init_sol = Solution([random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(-4, 4)])
        delta = 0.005
        max_iter = 800
        max_loops = 25
        min_progress = 0.0001
        tabu_list_max_length = 30
        M = 100
        # random weights
        a = round(random.uniform(0, 1), 2)
        weights = [a, round(1 - a, 2)]

    problem = MOO_Problem.FON
    constraints = [BoundaryConstraint([(-4, 4), (-4, 4), (-4, 4)])]
    search_alg = TabuSearchApriori(init_sol=init_sol, problem=problem, delta=delta, max_iter=max_iter,
                                   constraints=constraints, M=M, tabu_list_max_length=tabu_list_max_length,
                                   weights=weights, n_objectives=2, max_loops=max_loops, min_progress=min_progress)

    # running Tabu Search on MOO Problem FON
    search_history, termination_reason, last_iter, glob_sol = search_alg.search()
    final_sol = glob_sol
    print('Final solution is: ', final_sol)

    # plotting the objective space and the results (search history, etc...)
    global fig_num
    x_space = np.linspace(-4, 4, 50)  # same for all variables xi, for i=1,...,n
    f1 = []
    f2 = []
    p = 1./math.sqrt(3.)
    for x1 in x_space:
        for x2 in x_space:
            for x3 in x_space:
                sum1 = math.pow(x1 - p, 2) + math.pow(x2 - p, 2) + math.pow(x3 - p, 2)
                sum2 = math.pow(x1 + p, 2) + math.pow(x2 + p, 2) + math.pow(x3 + p, 2)
                f1.append(1 - math.exp(-sum1))
                f2.append(1 - math.exp(-sum2))
    plt.figure()
    plt.scatter(f1, f2, s=1.0, color='dodgerblue')

    # second part plots the Pareto front of the problem
    x_space = np.linspace(-p, p, 50)
    f1 = []
    f2 = []
    for x1 in x_space:
        x2 = x1
        x3 = x1
        sum1 = math.pow(x1 - p, 2) + math.pow(x2 - p, 2) + math.pow(x3 - p, 2)
        sum2 = math.pow(x1 + p, 2) + math.pow(x2 + p, 2) + math.pow(x3 + p, 2)
        f1.append(1 - math.exp(-sum1))
        f2.append(1 - math.exp(-sum2))
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='navy')

    # plotting the search history
    f1 = []
    f2 = []
    for sol in search_history:
        f1.append(sol.y[0])
        f2.append(sol.y[1])
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='red')
    plt.plot([f1[0]], [f2[0]], marker=">", markersize=15, color='darkgreen')  # starting position (init sol)
    plt.plot([f1[-1]], [f2[-1]], marker="s", markersize=15, color='darkgreen')  # final position (final sol)
    plt.plot(final_sol.y[0], final_sol.y[1], marker="*", markersize=15, color='gold')

    # add some plot labels
    # To every title I append the weights information as well, so firstly lets convert that to a string
    w_str = 'w=[' + str(weights[0]) + ", " + str(weights[1]) + "]"
    plt.title('Search history, Tabu Search, FON ' + w_str)
    plt.xlabel('f1(x1, x2)')
    plt.ylabel('f2(x1, x2)')
    plt.xlim(0, 1.2)
    plt.ylim(0, 1.2)
    plt.grid(True)

    # export data to json/txt in order to generate reports
    if save is True:
        # SAVE PROCEDURE
        # 1.) Navigate to appropriate test folder
        # 2.) Check number of files. Every test generates 2 files, .txt and .png. The test ID is equal to n_files / 2
        # 3.) Save the plot
        # 4.) Save the test data in json format to txt file

        folder = "/home/kemal/Programming/Python/Articulation/Tests/test_results_raw/TS/TS_apriori/FON"
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
        file_name = "FON_test_ID_" + str(test_ID)
        plt.savefig(file_name + '.png')
        save_test_to_file_vol(init_sol=init_sol, delta=delta, max_iter=max_iter, M=M, tabu_list_max_length=tabu_list_max_length,
                          weights=weights, max_loops=max_loops, min_progress=min_progress, final_sol=final_sol, seed_val=seed_val,
                          termination_reason=termination_reason, last_iter=last_iter, file_name=file_name)
    # plt.show()
    fig_num = fig_num + 1


# NOTE - For TNK I did not add any manual/standard tests. The reason is the complexity of determining
# whether the initial solutions are correct. I can add this later if required.
def TS_TNK_core(manual=False, std_ID=None, seed_val=0, toPlot=True, save=False):
    init_sol = delta = max_iter = M = tabu_list_max_length = weights = max_loops = min_progress = description = None
    # description not used for now.
    # Setting up the algorithm parameters
    if manual is True:
        # read the standard test setup from file
        # /Articulation/Tests/standardized_tests/TS/TNK/standard_i.txt, where i is = std_ID
        # load the json formatted data into a dictionary
        # Get the data from the the function load_standardized_test
        f_name = "/home/kemal/Programming/Python/Articulation/Tests/standardized_tests/TS/TNK/standard_" + str(std_ID) + ".txt"
        init_sol_x_vect, delta, max_iter, M, tabu_list_max_length, weights, max_loops, min_progress, description = load_standardized_test(f_name)
        init_sol = Solution(init_sol_x_vect)
    else:
        # Boundaries for problem TNK are xi € [0, pi]
        # NOTE - WHEN CHOOSING RANDOM INITIAL SOLUTION, IT MUST BE ENSURED THAT IT SATISFIES THE CONSTRAINTS!
        random.seed(seed_val)
        x1_init = 0
        x2_init = 0
        while 1:
            x1_init, x2_init = random.uniform(0, math.pi), random.uniform(0, math.pi)
            # check whether constraints are satisfied
            if math.isclose(x1_init, 0):
                c1_init = math.pow(x2_init, 2) - 1.1
            else:
                c1_init = math.pow(x1_init, 2) + math.pow(x2_init, 2) - 1 - 0.1*math.cos(16*math.atan(x2_init/x1_init))
            c2_init = math.pow(x1_init - 0.5, 2) + math.pow(x2_init - 0.5, 2) - 0.5
            if c1_init >= 0 >= c2_init:
                break
        init_sol = Solution([x1_init, x2_init])
        delta = 0.01
        max_iter = 500
        max_loops = 25
        min_progress = 0.0001
        tabu_list_max_length = 30
        M = 100
        # random weights
        a = round(random.uniform(0, 1), 2)
        weights = [a, round(1 - a, 2)]

    problem = MOO_Problem.TNK
    const1 = BoundaryConstraint([(0, math.pi), (0, math.pi)])
    const2 = GreaterThanConstraint(TNK_constraint_1)
    const3 = LessThanConstraint(TNK_constraint_2)
    constraints = [const1, const2, const3]
    search_alg = TabuSearchApriori(init_sol=init_sol, problem=problem, delta=delta, max_iter=max_iter,
                                   constraints=constraints, M=M, tabu_list_max_length=tabu_list_max_length,
                                   weights=weights, n_objectives=2, max_loops=max_loops, min_progress=min_progress)

    # running Tabu Search on MOO Problem TNK
    search_history, termination_reason, last_iter, glob_sol = search_alg.search()
    final_sol = glob_sol
    print('Final solution is: ', final_sol)

    # plotting the objective space and the results (search history, etc...)
    global fig_num
    x1_space = np.linspace(0, math.pi, 200)  # should be extra dense, because of the non-convex border
    x2_space = np.linspace(0, math.pi, 200)
    f1 = []
    f2 = []
    for x1 in x1_space:
        for x2 in x2_space:
            if math.isclose(x1, 0):
                c1 = math.pow(x2, 2) - 1.1
            else:
                c1 = math.pow(x1, 2) + math.pow(x2, 2) - 1 - 0.1 * math.cos(16 * math.atan(x2 / x1))
            c2 = math.pow(x1 - 0.5, 2) + math.pow(x2 - 0.5, 2) - 0.5
            if c1 >= 0 >= c2:
                f1.append(x1)
                f2.append(x2)
    plt.figure()
    plt.scatter(f1, f2, s=1.0, color='dodgerblue')

    # second part plots the Pareto front of the problem
    f1 = []
    f2 = []
    x1_space = np.linspace(0, math.pi, 1000)
    x2_space = np.linspace(0, math.pi, 1000)
    for x1 in x1_space:
        for x2 in x2_space:
            if math.isclose(x1, 0):
                c1 = math.pow(x2, 2) - 1.1
            else:
                c1 = math.pow(x1, 2) + math.pow(x2, 2) - 1 - 0.1 * math.cos(16 * math.atan(x2 / x1))
            c2 = math.pow(x1 - 0.5, 2) + math.pow(x2 - 0.5, 2) - 0.5
            if math.isclose(c1, 0, abs_tol=0.001) and 0.0 >= c2:
                f1.append(x1)
                f2.append(x2)

    f1_pom = []
    f2_pom = []
    # this additional nested for-loop is a brute-force method that filters out the actual Pareto optimums.
    # the reason for this is that even though the previous piece of code does find the curve which defines the
    # region border, its non convex and not the entire curve makes up the Pareto front.
    for i in range(len(f1)):
        p1, p2 = f1[i], f2[i]
        trig = False
        for j in range(len(f1)):
            if p1 - f1[j] > 0 and p2 - f2[j] > 0:
                trig = True
                break
            else:
                continue
        if not trig:
            f1_pom.append(p1)
            f2_pom.append(p2)

    f1 = f1_pom
    f2 = f2_pom
    plt.scatter(f1, f2, linewidth=3.5, linestyle='-', color='navy')

    # plotting the search history
    f1 = []
    f2 = []
    for sol in search_history:
        f1.append(sol.y[0])
        f2.append(sol.y[1])
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='red')
    plt.plot([f1[0]], [f2[0]], marker=">", markersize=15, color='darkgreen')  # starting position (init sol)
    plt.plot([f1[-1]], [f2[-1]], marker="s", markersize=15, color='darkgreen')  # final position (final sol)
    plt.plot(final_sol.y[0], final_sol.y[1], marker="*", markersize=15, color='gold')

    # add some plot labels
    # To every title I append the weights information as well, so firstly lets convert that to a string
    w_str = 'w=[' + str(weights[0]) + ", " + str(weights[1]) + "]"
    plt.title('Search history, Tabu Search, TNK ' + w_str)
    plt.xlabel('f1(x1) = x1')
    plt.ylabel('f2(x2) = x2')
    plt.xlim(0, 1.4)
    plt.ylim(0, 1.4)
    plt.grid(True)

    # export data to json/txt in order to generate reports
    if save is True:
        # SAVE PROCEDURE
        # 1.) Navigate to appropriate test folder
        # 2.) Check number of files. Every test generates 2 files, .txt and .png. The test ID is equal to n_files / 2
        # 3.) Save the plot
        # 4.) Save the test data in json format to txt file

        folder = "/home/kemal/Programming/Python/Articulation/Tests/test_results_raw/TS/TS_apriori/TNK"
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
        file_name = "TNK_test_ID_" + str(test_ID)
        plt.savefig(file_name + '.png')
        save_test_to_file_vol(init_sol=init_sol, delta=delta, max_iter=max_iter, M=M, tabu_list_max_length=tabu_list_max_length,
                          weights=weights, max_loops=max_loops, min_progress=min_progress, final_sol=final_sol, seed_val=seed_val,
                          termination_reason=termination_reason, last_iter=last_iter, file_name=file_name)
    # plt.show()
    fig_num = fig_num + 1


# CONSTRAINTS ARE PROBABLY WRONG.
def TS_OSY_core(manual=False, std_ID=None, seed_val=0, toPlot=True, save=False):
    init_sol = delta = max_iter = M = tabu_list_max_length = weights = max_loops = min_progress = description = None
    # description not used for now.
    # Setting up the algorithm parameters
    if manual is True:
        # read the standard test setup from file
        # /Articulation/Tests/standardized_tests/TS/OSY/standard_i.txt, where i is = std_ID
        # load the json formatted dataa into a dictionary
        # Get the data from the the function load_standardized_test
        f_name = "/home/kemal/Programming/Python/Articulation/Tests/standardized_tests/TS/OSY/standard_" + str(std_ID) + ".txt"
        init_sol_x_vect, delta, max_iter, M, tabu_list_max_length, weights, max_loops, min_progress, description = load_standardized_test(f_name)
        init_sol = Solution(init_sol_x_vect)
    else:
        x_tmp = []
        while(1):
            x_tmp = [random.uniform(0, 10), random.uniform(0, 10), random.uniform(1, 5), random.uniform(0, 6), random.uniform(1, 5), random.uniform(0, 10)]
            if OSY_constraint_1(x_tmp) > 0 and OSY_constraint_2(x_tmp) > 0 and OSY_constraint_3(x_tmp) > 0 and OSY_constraint_4(x_tmp) > 0 and OSY_constraint_5(x_tmp) > 0 and OSY_constraint_6(x_tmp) > 0:
                break
        print('found initial point')
        # Boundaries for problem OSY are x1,2,6 € [0, 10], x3,5 € [1, 5], x4 € [0, 6]
        random.seed(seed_val)
        init_sol = Solution(x_tmp)
        delta = 0.02
        max_iter = 100
        max_loops = 20
        min_progress = 0.001
        tabu_list_max_length = 20
        M = 2000
        # random weights
        a = round(random.uniform(0, 1), 2)
        weights = [a, round(1 - a, 2)]

    problem = MOO_Problem.OSY
    const0 = BoundaryConstraint([(0, 10), (0, 10), (1, 5), (0, 6), (1, 5), (0, 10)])
    const1 = GreaterThanConstraint(OSY_constraint_1)
    const2 = GreaterThanConstraint(OSY_constraint_2)
    const3 = GreaterThanConstraint(OSY_constraint_3)
    const4 = GreaterThanConstraint(OSY_constraint_4)
    const5 = GreaterThanConstraint(OSY_constraint_5)
    const6 = GreaterThanConstraint(OSY_constraint_6)
    constraints = [const0, const1, const2, const3, const4, const5, const6]


    search_alg = TabuSearchApriori(init_sol=init_sol, problem=problem, delta=delta, max_iter=max_iter,
                                   constraints=constraints, M=M, tabu_list_max_length=tabu_list_max_length,
                                   weights=weights, n_objectives=2, max_loops=max_loops, min_progress=min_progress)

    # running Tabu Search on MOO Problem OSY
    search_history, termination_reason, last_iter, glob_sol = search_alg.search()
    final_sol = glob_sol
    print('Final solution is: ', final_sol)

    # plotting the objective space and the results (search history, etc...)
    global fig_num
    f1 = []
    f2 = []
    random.seed(0)
    while len(f1) < 1e5:    # 100k sample points
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        x3 = random.uniform(1, 5)
        x4 = random.uniform(0, 6)
        x5 = random.uniform(0, 5)
        x6 = random.uniform(0, 10)
        if x1 + x2 - 2 < 0:
            continue
        if 6 - x1 - x2 < 0:
            continue
        if 2 + x1 - x2 < 0:
            continue
        if 2 - x1 + 3 * x2 < 0:
            continue
        if 4 - math.pow(x3 - 3, 2) - x4 < 0:
            continue
        if math.pow(x5 - 3, 2) + x6 - 4 < 0:
            continue
        f1.append(
            -25 * math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - math.pow(x4 - 4, 2) - math.pow(
                x5 - 1, 2))
        f2.append(x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 + x6 ** 2)

    plt.figure()
    plt.scatter(f1, f2, s=1.0, color='dodgerblue')


    # second part plots the Pareto front of the problem
    f1 = []
    f2 = []
    x4, x6 = 0, 0
    # region AB
    x1, x2, x5 = 5, 1, 5
    x3_space = np.linspace(1, 5, 100)
    for x3 in x3_space:
        f1.append(-25 * math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1 ** 2 + x2 ** 2 + x3 ** 2 + x5 ** 2)

    # region BC
    x1, x2, x5 = 5, 1, 1
    x3_space = np.linspace(1, 5, 100)
    for x3 in x3_space:
        f1.append(-25 * math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1 ** 2 + x2 ** 2 + x3 ** 2 + x5 ** 2)

    # region CD
    x1_space = np.linspace(4.056, 5, 50)
    x3, x5 = 1, 1
    for x1 in x1_space:
        x2 = (x1 - 2.) / 3.
        f1.append(-25 * math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1 ** 2 + x2 ** 2 + x3 ** 2 + x5 ** 2)

    # region DE
    x1, x2, x5 = 0, 2, 1
    x3_space = np.linspace(1, 3.732, 100)
    for x3 in x3_space:
        f1.append(-25 * math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1 ** 2 + x2 ** 2 + x3 ** 2 + x5 ** 2)

    # region EF
    x1_space = np.linspace(0, 1, 50)
    x3, x5 = 1, 1
    for x1 in x1_space:
        x2 = 2 - x1
        f1.append(-25 * math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1 ** 2 + x2 ** 2 + x3 ** 2 + x5 ** 2)

    plt.scatter(f1, f2, linewidth=3.5, linestyle='-', color='navy')

    # plotting the search history
    f1 = []
    f2 = []
    for sol in search_history:
        f1.append(sol.y[0])
        f2.append(sol.y[1])
    plt.plot(f1, f2, linewidth=3.5, linestyle='-', color='red')
    plt.plot([f1[0]], [f2[0]], marker=">", markersize=15, color='darkgreen')  # starting position (init sol)
    plt.plot([f1[-1]], [f2[-1]], marker="s", markersize=15, color='darkgreen')  # final position (final sol)
    plt.plot(final_sol.y[0], final_sol.y[1], marker="*", markersize=15, color='gold')
    # add some plot labels
    # To every title I append the weights information as well, so firstly lets convert that to a string
    w_str = 'w=[' + str(weights[0]) + ", " + str(weights[1]) + "]"
    plt.title('Search history, Tabu Search, OSY ' + w_str)
    plt.xlabel('f1(x1, x2, x3, x4, x5, x6)')
    plt.ylabel('f2(x1, x2, x3, x4, x5, x6)')
    plt.xlim(-300, 0)
    plt.ylim(0, 80)
    plt.grid(True)

    # export data to json/txt in order to generate reports
    if save is True:
        # SAVE PROCEDURE
        # 1.) Navigate to appropriate test folder
        # 2.) Check number of files. Every test generates 2 files, .txt and .png. The test ID is equal to n_files / 2
        # 3.) Save the plot
        # 4.) Save the test data in json format to txt file

        folder = "/home/kemal/Programming/Python/Articulation/Tests/test_results_raw/TS/TS_apriori/OSY"
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
        file_name = "OSY_test_ID_" + str(test_ID)
        plt.savefig(file_name + '.png')
        save_test_to_file_vol(init_sol=init_sol, delta=delta, max_iter=max_iter, M=M, tabu_list_max_length=tabu_list_max_length,
                          weights=weights, max_loops=max_loops, min_progress=min_progress, final_sol=final_sol, seed_val=seed_val,
                          termination_reason=termination_reason, last_iter=last_iter, file_name=file_name)
    # plt.show()
    fig_num = fig_num + 1




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
    # print(args)
    # print(optlist)
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

    if prob == 'IM1':
        TS_IM1_core(manual=man, std_ID=std_ID_, seed_val=seed_val_, toPlot=True, save=save)
    elif prob == 'BK1':
        TS_BK1_core(manual=man, std_ID=std_ID_, seed_val=seed_val_, toPlot=True, save=save)
    elif prob == 'SCH1':
        TS_SCH1_core(manual=man, std_ID=std_ID_, seed_val=seed_val_, toPlot=True, save=save)
    elif prob == 'FON':
        TS_FON_core(manual=man, std_ID=std_ID_, seed_val=seed_val_, toPlot=True, save=save)
    elif prob == 'TNK':
        TS_TNK_core(manual=man, std_ID=std_ID_, seed_val=seed_val_, toPlot=True, save=save)
    elif prob == 'OSY':
        TS_OSY_core(manual=man, std_ID=std_ID_, seed_val=seed_val_, toPlot=True, save=save)
    else:
        print('Error! Did not find benchmark problem name! Exiting with value 10.')
        sys.exit(10)


if __name__ == '__main__':
    main()

