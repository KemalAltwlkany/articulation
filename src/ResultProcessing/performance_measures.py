import os as os
import pickle as pickle
import copy as copy
import pandas as pd
import numpy as np


def compute_runtime(runtimes, alternatives, objectives, data):
    # runtimes = np.sort(runtimes)
    return [np.average(runtimes), np.min(runtimes), np.max(runtimes), np.median(runtimes), np.std(runtimes)]

def compute_euclidean_distance(runtimes, alternatives, objectives, data):
    distances = -1*np.ones(len(alternatives))
    for i in range(len(alternatives)):
        distances[i] = np.min(np.linalg.norm(data['x'] - alternatives[i], axis=1))
        if distances[i] < 0:
            print('ERROR - A DISTANCE IS NEGATIVE!')
    return [np.average(distances), np.min(distances), np.max(distances), np.median(distances), np.std(distances)]

def compute_generational_distance(runtimes, alternatives, objectives, data):
    print(objectives[0:2])
    print(data['pareto_f1'][0:2])
    print(data['pareto_f2'][0:2])
    print(type(data['pareto_f1']))

    # <np.array> of <np.arrays>. Basically array containing (f1,f2) pairs which € to the pareto front.
    pfs = np.stack((data['pareto_f1'], data['pareto_f2']), axis=1)
    print(pfs[0:2])
    distances = -1*np.ones(len(objectives))
    for i in range(len(objectives)):
        distances[i] = np.min(np.linalg.norm(pfs - objectives[i], axis=1))
        if distances[i] < 0:
            print('ERROR - A DISTANCE IS NEGATIVE!')

    # Parameter denoted as p by Veldhuizen and Deb is taken to be p=2.
    GD = np.sqrt(np.sum(np.square(distances))) / len(objectives)
    return [GD, np.average(distances), np.min(distances), np.max(distances), np.median(distances), np.std(distances)]

def compute_spacing(runtimes, alternatives, objectives, data):
    # For every solution, i.e. for every (f1_i, f2_i) pair in objectives
    # it is necessary to compute d_i, which is the Manhattan distance of the other closest (f1_j, f2_j) pair in objectives.
    print(type(objectives))
    print(objectives[0:2])
    x = np.array(objectives)
    print(type(x))
    print(x[0:2])
    x_masked = np.ma.masked_array(x, mask=False)

    distances = -1 * np.ones(len(x))
    for i in range(len(distances)):
        x_masked.mask[i] = True
        distances[i] = np.ma.min(np.ma.sum(np.ma.abs(x_masked - x[i]), axis=1))
        x_masked.mask[i] = False
        if distances[i] < 0:
            print('ERROR - A DISTANCE IS NEGATIVE!')

    # kept for illustration purposes, since spacing is literally the standard deviation of the distances.
    #d_ = np.average(distances)
    #spacing = np.sqrt(np.sum(np.square(distances - d_))/len(distances))
    return [np.average(distances), np.min(distances), np.max(distances), np.median(distances), np.std(distances)]


def compute_spread(runtimes, alternatives, objectives, data):
    # Extremal values of (f1, f2) in order: BK1, IM1, SCH1, FON, TNK, OSY
    #0.0 50.0
    #50.0 0.0
    #2.0 4.0
    #4.0 1.0
    #0.0 4.0
    #4.0 0.0
    #0.9816843611112658 0.0
    #0.0 0.9816843611112658
    #0.044004299299907054 1.0372441977835234
    #1.0372441977835234 0.044004299299907054
    #-274.0 76.0
    #-42.0 4.0
    f1_e = np.array([data['pareto_f1'][0], data['pareto_f2'][0]])
    f2_e = np.array([data['pareto_f1'][-1], data['pareto_f2'][-1]])
    # 1st componente of the metric is the same as when computing spacing.
    x = np.array(objectives)
    x_masked = np.ma.masked_array(x, mask=False)

    distances = -1 * np.ones(len(x))
    for i in range(len(distances)):
        x_masked.mask[i] = True
        distances[i] = np.ma.min(np.ma.sum(np.ma.abs(x_masked - x[i]), axis=1))
        x_masked.mask[i] = False
        if distances[i] < 0:
            print('ERROR - A DISTANCE IS NEGATIVE!')

    # 2nd component of spread metric
    d1_distances = np.sum(np.abs(x - f1_e), axis=1)
    d2_distances = np.sum(np.abs(x - f2_e), axis=1)
    d1_e = np.min(d1_distances)
    d2_e = np.min(d2_distances)
    spread = d1_e + d2_e + np.sum(distances - np.average(distances))
    spread = spread / (d1_e + d2_e + len(distances)*np.average(distances))

    return [spread, d1_e, d2_e, np.average(distances), np.min(distances), np.max(distances), np.median(distances), np.std(distances),
            np.average(d1_distances), np.max(d1_distances), np.median(d1_distances), np.std(d1_distances),
            np.average(d2_distances), np.max(d2_distances), np.median(d2_distances), np.std(d2_distances)]


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
    for p in file_names:
        file = open(load_path + p, 'rb')
        d = pickle.load(file)
        runtimes.append(d['time_elapsed'])
        alternatives.append(copy.deepcopy(d['global_best_sol'].get_x()))
        objectives.append(copy.deepcopy(d['global_best_sol'].get_y()))
        file.close()


    # determine which performance measures are to be computed
    perf_measures = ['runtime', 'euclidean_distance', 'generational_distance', 'spacing', 'spread']
    if which_PM is not 'all':
        perf_measures = [which_PM]

    # load precomputed data
    file = open('/home/kemal/Programming/Python/Articulation/data/precomputed_data/' + problem_name + '_data.pickle', 'rb')
    data = pickle.load(file)
    file.close()
    alternatives = np.array(alternatives)
    for PM in perf_measures:
        df_filename = '/home/kemal/Programming/Python/Articulation/data/performance_measures/PM_' + PM + '.csv'
        df = pd.read_csv(df_filename, index_col='entry')  # open appropriate .csv file
        df.loc[len(df.index)] = [articulation_type, problem_name] + function_mapping[PM](runtimes, alternatives, objectives, data)  # compute appropriate performance measure
        #print(function_mapping[PM](runtimes, alternatives, objectives, data))
        df.to_csv(df_filename)



if __name__ == '__main__':
    #for art_type in ['aposteriori', 'apriori', 'progressive']:
    #for prob_name in ['BK1', 'IM1', 'SCH1', 'FON', 'TNK', 'OSY']:
    #    compute_performance_measures(articulation_type='apriori', problem_name=prob_name, which_PM='spread')
    compute_performance_measures(articulation_type='apriori', problem_name='IM1', which_PM='spread')

