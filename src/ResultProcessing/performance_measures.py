import os as os
import pickle as pickle
import copy as copy
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def compute_runtime(runtimes, alternatives, objectives, data):
    """
    Function computes average, min, max, median and std of runtimes.
    Does not modify runtimes.
    Other function arguments are kept only to ease out the coding, they are never used.
    :param runtimes: list of runtimes per test run.
    :param alternatives: Not used.
    :param objectives: Not used.
    :param data: Not used.
    :return:
    """
    return [np.average(runtimes), np.min(runtimes), np.max(runtimes), np.median(runtimes), np.std(runtimes)]

def compute_euclidean_distance(runtimes, alternatives, objectives, data):
    """
    Function computes the Euclidean distance between every optimum found (alternatives) and its nearest neighbor from
    the Pareto set. Returns the avg, min, max, median and std of the Euclidean distance.
    :param runtimes: Not used.
    :param alternatives: <list> of alternatives found per test run.
    :param objectives: Not used.
    :param data: <dict> loaded from the precomputed files. Contains <np.array> of Pareto set under key 'x'.
    :return:
    """
    distances = -1*np.ones(len(alternatives))
    for i in range(len(alternatives)):
        #distances[i] = np.min(np.abs(data['x'] - alternatives[i]))  # only for SCH1
        distances[i] = np.min(np.linalg.norm(data['x'] - alternatives[i], axis=1))
        if distances[i] < 0:
            print('ERROR - A DISTANCE IS NEGATIVE!')
    return [np.average(distances), np.min(distances), np.max(distances), np.median(distances), np.std(distances)]

def compute_generational_distance(runtimes, alternatives, objectives, data):
    """
    Computes the generational distance. This is a scalar performance measure, which is computed along all the
    solutions found.
    :param runtimes: Not used.
    :param alternatives: Not used.
    :param objectives: <list> of <np.array> containing (f1, f2) for every optimum found.
    :param data: <dict> loaded from the precomputed files. Contains <np.array> of Pareto front under keys 'pareto_f1' and 'pareto_f2'
    :return:
    """
    # <np.array> of <np.arrays>. Basically array containing (f1,f2) pairs which € to the pareto front.
    pfs = np.stack((data['pareto_f1'], data['pareto_f2']), axis=1)
    distances = -1*np.ones(len(objectives))
    for i in range(len(objectives)):
        distances[i] = np.min(np.linalg.norm(pfs - objectives[i], axis=1))
        if distances[i] < 0:
            print('ERROR - A DISTANCE IS NEGATIVE!')

    # Parameter denoted as p by Veldhuizen and Deb is taken to be p=2.
    GD = np.sqrt(np.sum(np.square(distances))) / len(objectives)
    return [GD, np.average(distances), np.min(distances), np.max(distances), np.median(distances), np.std(distances)]

def compute_spacing(runtimes, alternatives, objectives, data, visualize=False):
    """
    Calculates the spacing in the "generation". The generation is defined as the set of all Pareto optimal solutions found,
    i.e. the set of all solutions from every test run. Spacing is practically the standard deviation of the obtained Pareto front.
    :param runtimes: Not used.
    :param alternatives: Not used.
    :param objectives: <list> of <np.array> containing (f1, f2) for every optimum found.
    :param data: Not used.
    :return:
    """
    # For every solution, i.e. for every (f1_i, f2_i) pair in objectives
    # it is necessary to compute d_i, which is the Manhattan distance of the other closest (f1_j, f2_j) pair in objectives.
    x = np.array(objectives)
    x_masked = np.ma.masked_array(x, mask=False)
    distances = -1 * np.ones(len(x))
    for i in range(len(distances)):
        x_masked.mask[i] = True
        distances[i] = np.ma.min(np.ma.sum(np.ma.abs(x_masked - x[i]), axis=1))
        x_masked.mask[i] = False
        if distances[i] < 0:
            print('ERROR - A DISTANCE IS NEGATIVE!')

    # kept for illustration purposes, since spacing is literally the standard deviation of the distances.
    # d_ = np.average(distances)
    # spacing = np.sqrt(np.sum(np.square(distances - d_))/len(distances))

    # Visualization part
    if visualize is True:
        # plot the precomputed data
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(name='K samples of PF', x=x[:, 0], y=x[:, 1], mode='markers',
                       marker=dict(color='red'), marker_size=5))


        fig.update_layout(title={
            'text': str('<b>' + 'Spacing measure' + '</b>'),
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            xaxis_title='f1', yaxis_title='f2', legend_title='<b>Legend:</b>')
        fig.show()

    return [np.average(distances), np.min(distances), np.max(distances), np.median(distances), np.std(distances)]

def compute_spread(runtimes, alternatives, objectives, data):
    """
    This metric requires the knowledge of the extremal Pareto front points for every test problem. These are shown
    in the comment bellow, but are basically the first and last values of the data['pareto_f1'] and data['pareto_f2'] dict.
    :param runtimes: Not used.
    :param alternatives: Not used.
    :param objectives: <list> of <np.array> containing (f1, f2) for every optimum found.
    :param data: <dict> loaded from the precomputed files. Contains <np.array> of Pareto front under keys 'pareto_f1' and 'pareto_f2'
    :return:
    """
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

def compute_chi_square_like(runtime, alternatives, objectives, data, visualize=False):
    """
    Computes the Chi-square-like deviation measure as per Deb. Two parameters need to be selected in order to compute this
    PM. These are:
    1.) K - the number of samples/Pareto front solutions considered. These are selected as the 5 a priori/progressive aspiration
    levels, specified by the DM
    2.) eps - defines the niche of every of those K samples.
    These parameters are hardcoded into a dictionary, as the test problems are fixed, so it is logical to fix them as well.
    :param visualize:
    :param runtime: Not used.
    :param alternatives: Not used.
    :param objectives: <list> of <np.array> containing (f1, f2) for every optimum found.
    :param data: <dict> loaded from the precomputed files. Contains <np.array> of Pareto front under keys 'pareto_f1' and 'pareto_f2'
    :return:
    """
    # The number of points for which the Chi-squared-like deviation measure is computed
    # is set to 5. These 5 points correspond to the aspiration levels (goals)
    # set in a priori and progressive articulation.

    objs = np.array(objectives)
    # Take 5 uniformly sampled points from the Pareto front
    f1_samps = np.concatenate((data['pareto_f1'][::len(data['pareto_f1'])//4], np.array([data['pareto_f1'][-1]])))
    f2_samps = np.concatenate((data['pareto_f2'][::len(data['pareto_f2'])//4], np.array([data['pareto_f2'][-1]])))

    # ONLY FOR TNK
    # f1_samps = np.delete(f1_samps, len(f1_samps)-1)
    # f2_samps = np.delete(f2_samps, len(f2_samps)-1)

    five_points = np.stack((f1_samps, f2_samps), axis=1)

    print(five_points)
    # Need to find optimal parameter eps, by computing distances between each of the five points and its nearest neighbor
    # eps will be the average distance of all distances between closest neighbors
    inds = np.array([True] * len(five_points))
    dists = -1 * np.ones(len(five_points))
    for i in range(len(five_points)):
        inds[i] = False
        dists[i] = np.min(np.sqrt(np.sum(np.square(five_points[i] - five_points[inds]), axis=1)))
        inds[i] = True
    eps = np.min(dists)


    # parameter names and labels are denoted with respect to my master thesis
    q = len(objs)
    K = len(five_points)
    n_i = np.array([q/K]*K + [0])
    sigma_i = n_i[0] * (1.0 - n_i[0]/q)
    sigma_K_1 = q * (1.0 - 1.0/K)
    sigma = np.array([sigma_i]*K + [sigma_K_1])

    # for each of the 5 pareto optimal pts, check which obtained solutions are in its niche
    n_calc = np.zeros(K + 1)
    x = np.array(five_points)
    for i in range(len(x)):
        distances = np.sqrt(np.sum(np.square(objs - x[i]), axis=1))
        indices = distances < eps
        n_calc[i] = np.count_nonzero(indices)
        objs[indices] = np.array([[500, 500]])
    if np.sum(n_calc) > q:
        print('ERROR! ALGORITHM ESTIMATED TWO SOLS IN NICHE OF DIFFERENT FRONT SAMPLES!')
    n_calc[-1] = q - np.sum(n_calc)
    print('Computed: ', n_calc)
    print('Expected: ', n_i)
    CSLDM = np.sqrt(np.sum(np.divide(np.square(n_calc - n_i), sigma)))

    # Visualization part
    if visualize is True:
        # plot the precomputed data
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(name='K samples of PF', x=five_points[:, 0], y=five_points[:, 1], opacity=0.5, mode='markers',
                       marker=dict(color='red'), marker_size=20))
        objs_plot = np.array(objectives)
        fig.add_trace(
            go.Scatter(name='Obtained solutions', x=objs_plot[:, 0], y=objs_plot[:, 1], opacity=0.5, mode='markers',
                       marker=dict(color='blue'), marker_size=10))

        fig.update_layout(title={
            'text': str('<b>' + 'Chi-square-like deviation measure' + '</b>'),
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
            xaxis_title='f1', yaxis_title='f2', legend_title='<b>Legend:</b>')
        fig.show()
    return [CSLDM, n_i[0], sigma_i, sigma_K_1]

def compute_performance_measures(articulation_type=None, problem_name=None, which_PM=None, save=False):
    function_mapping = dict(
        runtime=compute_runtime,
        euclidean_distance=compute_euclidean_distance,
        generational_distance=compute_generational_distance,
        spacing=compute_spacing,
        spread=compute_spread,
        chi_square=compute_chi_square_like
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
        df.loc[len(df.index)] = [articulation_type, problem_name, len(runtimes)] + function_mapping[PM](runtimes, alternatives, objectives, data)  # compute appropriate performance measure
        #print(function_mapping[PM](runtimes, alternatives, objectives, data))
        if save is True:
            df.to_csv(df_filename)



if __name__ == '__main__':
    for art_type in ['aposteriori', 'apriori', 'progressive']:
        #for prob_name in ['BK1', 'IM1', 'SCH1', 'FON', 'TNK', 'OSY']:
        for PM in ['runtime', 'euclidean_distance', 'generational_distance', 'spacing', 'spread', 'chi_square']:
            compute_performance_measures(articulation_type=art_type, problem_name='OSY', which_PM=PM, save=True)
    #compute_performance_measures(articulation_type='aposteriori', problem_name='SCH1', which_PM='chi_square', save=False)
    #chi_square_like(1, 2, 3, 4, 5)
    #print('Not active.')

