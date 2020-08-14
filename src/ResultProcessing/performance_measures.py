import os as os
import pickle as pickle
import copy as copy
import pandas as pd
import numpy as np


def compute_runtime(runtimes, alternatives, objectives):
    # runtimes = np.sort(runtimes)
    return [np.average(runtimes), np.min(runtimes), np.max(runtimes), np.median(runtimes), np.std(runtimes)]

def compute_euclidean_distance(sols):
    pass

def compute_generational_distance(sols):
    pass

def compute_spacing(sols):
    pass

def compute_spread(sols):
    pass

def compute_performance_measures(articulation_type=None, problem_name=None, which_PM=None, save=False):
    function_mapping = dict(
        runtime=compute_runtime,
        euclidean_distance=compute_euclidean_distance,
        generational_distance=compute_generational_distance,
        spacing=compute_spacing,
        spread=compute_spread
    )
    load_path = '/home/kemal/Programming/Python/Articulation/data/pickles/' + articulation_type + '/' + problem_name + '/'
    os.chdir(load_path)
    file_names = os.listdir()
    objectives = []
    alternatives = []
    runtimes = []
    #print(file_names[0])
    for p in file_names:
        file = open(load_path + p, 'rb')
        d = pickle.load(file)
        runtimes.append(d['time_elapsed'])
        alternatives.append(copy.deepcopy(d['global_best_sol'].get_x()))
        objectives.append(copy.deepcopy(d['global_best_sol'].get_y()))
        file.close()
    # print(len(alternatives))
    # print(alternatives[0])
    # print(len(objectives))
    # print(objectives[0])
    # print(len(runtimes))
    # print(runtimes[0])

    # determine which performance measures are to be computed
    perf_measures = ['runtime', 'euclidean_distance', 'generational_distance', 'spacing', 'spread']
    if which_PM is not 'all':
        perf_measures = [which_PM]

    for PM in perf_measures:
        df_filename = '/home/kemal/Programming/Python/Articulation/data/performance_measures/PM_' + PM + '.csv'
        df = pd.read_csv(df_filename, index_col='entry')  # open appropriate .csv file
        df.loc[len(df.index)] = [articulation_type, problem_name] + function_mapping[PM](runtimes, alternatives, objectives)  # compute appropriate performance measure
        df.to_csv(df_filename)



if __name__ == '__main__':
    #for art_type in ['aposteriori', 'apriori', 'progressive']:
    #    for prob_name in ['BK1', 'IM1', 'SCH1', 'FON', 'TNK', 'OSY']:
    #        compute_performance_measures(articulation_type='apriori', problem_name='BK1', which_PM='runtime')
    compute_performance_measures(articulation_type='apriori', problem_name='IM1', which_PM='runtime')

