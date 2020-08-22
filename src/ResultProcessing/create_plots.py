import os as os
import pickle as pickle
import plotly.graph_objects as go
from itertools import product
from src.PreferenceArticulation.BenchmarkObjectives import *


def plot_search_results(articulation_type, benchmark_problem, search_results, title='Test', x_label='x', y_label='y', show=False, save=False, save_options=None):
    # load precomputed data (search space and Pareto front)
    load_path = '/home/kemal/Programming/Python/Articulation/data/precomputed_data/'
    file = open(load_path + benchmark_problem + '_data.pickle', 'rb')
    data = pickle.load(file)
    file.close()

    # plot the precomputed data
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name='Objective space', x=data['search_f1'], y=data['search_f2'], mode='markers',
                   marker=dict(color='orange'), marker_size=5))
    fig.add_trace(
        go.Scatter(name='Pareto front', x=data['pareto_f1'], y=data['pareto_f2'], mode='markers',
                   marker=dict(color='blue'),
                   marker_size=5))

    # extract search history information from "search_results"
    search_history = search_results['search_history']
    global_best_sol = search_results['global_best_sol']

    # plot the search path
    arr = np.array([i.get_y() for i in search_history])
    f1, f2 = arr[:, 0], arr[:, 1]
    fig.add_trace(go.Scatter(name='Search path', x=f1, y=f2, mode='lines', line=dict(color='red', width=4)))

    # mark the starting and finish points and the global best solution
    f1, f2 = [arr[0, 0]], [arr[0, 1]]
    fig.add_trace(go.Scatter(name='Initial point', x=f1, y=f2, mode='markers', marker_symbol='triangle-right',
                             marker_color='lime', marker_size=25))

    f1, f2 = [arr[-1, 0]], [arr[-1, 1]]
    fig.add_trace(
        go.Scatter(name='Finish point', x=f1, y=f2, mode='markers', marker_symbol='square', marker_color='lime',
                   marker_size=28))

    # Update 10.08.2020. - added plotting the goals/aspiration levels if specified articulation type is a priori
    if articulation_type == 'apriori' or articulation_type == 'progressive':
        aspiration_levels = search_results['aspirations']
        z1, z2 = [aspiration_levels[0]], [aspiration_levels[1]]
        item_name = 'Aspiration vector'
        if articulation_type is 'apriori':
            item_name = 'Goal'
        fig.add_trace(
            go.Scatter(name=item_name, x=z1, y=z2, mode='markers', marker_symbol='hexagon', marker_color='fuchsia',
                       marker_size=22))

    # Update 13.08.2020. - added plotting the best solutions per repetition if specified articulation type is progressive
    if articulation_type == 'progressive':
        best_sols = search_results['best_sols_history']
        f1, f2 = [], []
        for sol in best_sols:
            y = sol.get_y()
            f1.append(y[0])
            f2.append(y[1])
        fig.add_trace(
            go.Scatter(name='Best solution path', x=f1, y=f2, mode='markers', marker_symbol='star-diamond', marker_color='darkmagenta',
                       marker_size=22))

    f1, f2 = [global_best_sol.get_y()[0]], [global_best_sol.get_y()[-1]]
    fig.add_trace(go.Scatter(name='Optimum', x=f1, y=f2, mode='markers', marker_symbol='star', marker_color='yellow',
                             marker_size=22))

    fig.update_layout(title={
        'text': str('<b>' + title + '</b>'),
        'y': 0.9,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        xaxis_title=x_label, yaxis_title=y_label, legend_title='<b>Legend:</b>')

    if show is True:
        fig.show()

    if save is True:
        os.chdir(save_options['path'])
        fig.write_image(save_options['path'] + save_options['name'])


# Can/should be modified so that I can specify whether to plot one particular test (or particular tests) instead or all.
def create_plots(articulation_type, benchmark_problem, extension='.png', n_samples=50, x_label='f1(x1, x2)', y_label='f2(x1, x2)'):
    load_path = '/home/kemal/Programming/Python/Articulation/data/pickles/' + articulation_type + '/' + benchmark_problem + '/'
    save_path = '/home/kemal/Programming/Python/Articulation/data/txts_and_plots/' + articulation_type + '/' + benchmark_problem + '/' + 'plots' + '/'
    # Some mappings for pretty outputs
    name_mappings = {
        'aposteriori': 'A posteriori',
        'progressive': 'Progressive',
        'apriori': 'A priori'
    }

    os.chdir(load_path)
    file_names = os.listdir()
    # Each iteration one pickled test is read, and a plot is created.
    for p in file_names:
        file = open(load_path + p, 'rb')
        d = pickle.load(file)
        file.close()

        search_results = dict(
            search_history=d['search_history'],
            global_best_sol=d['global_best_sol']
        )

        # Update - 10.08.2020. figured I need the aspiration levels passed as well. Might add the weights while I'm at it too.
        if articulation_type is 'apriori' or articulation_type is 'progressive':
            search_results['aspirations'] = d['aspirations']

        # Update - 13.08.2020. figured I need results from every search run in progressive articulation.
        if articulation_type is 'progressive':
            search_results['best_sols_history'] = d['best_sols_history']

        save_options = dict(
            path=save_path,
            name=str(d['test_ID'] + extension)
        )
        plot_search_results(articulation_type, benchmark_problem, search_results, title=str(name_mappings[articulation_type] + ' ' + benchmark_problem),
                            x_label=x_label, y_label=y_label, show=False, save=True,
                            save_options=save_options)


if __name__ == '__main__':
    # A POSTERIORI TESTS
    #create_plots('aposteriori', 'BK1')
    #create_plots('aposteriori', 'IM1')
    # create_plots('aposteriori', 'SCH1')
    # create_plots('aposteriori', 'FON')
    # create_plots('aposteriori', 'TNK')
    # create_plots('aposteriori', 'OSY')

    # A PRIORI TESTS
    #create_plots('apriori', 'BK1')
    #create_plots('apriori', 'IM1')
    #create_plots('apriori', 'SCH1')
    #create_plots('apriori', 'FON')
    # create_plots('apriori', 'TNK')
    #create_plots('apriori', 'OSY')

    # PROGRESSIVE TESTS
    #create_plots('progressive', 'BK1')
    #create_plots('progressive', 'IM1')
    #create_plots('progressive', 'SCH1')
    #create_plots('progressive', 'FON')
    create_plots('progressive', 'TNK')
    #create_plots('progressive', 'OSY')

    #print('Not active.')
