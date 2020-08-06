import os as os
import numpy as np
import pickle as pickle
from src.PreferenceArticulation.Solution import Solution
from src.PreferenceArticulation.BenchmarkObjectives import *
from src.TabuSearch.weighting_method import AposterioriWeightingMethod
import plotly.graph_objects as go


def plot_BK1_objective_space(n_samples=100, show=True):
    # The objective space - a few samples
    x1_range = np.linspace(-5, 10, n_samples)
    x2_range = np.linspace(-5, 10, n_samples)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    f1 = x1 ** 2 + x2 ** 2
    f2 = (x1 - 5) ** 2 + (x2 - 5) ** 2
    f1 = np.reshape(f1, f1.size)
    f2 = np.reshape(f2, f2.size)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name='Objective space', x=f1, y=f2, mode='markers', marker=dict(color='orange'), marker_size=5))

    # The Pareto front
    x1_range = np.linspace(0, 5, n_samples // 10)
    x2_range = np.linspace(0, 5, n_samples // 10)
    # x1, x2 = np.meshgrid(x1_range, x2_range)
    x1, x2 = x1_range, x2_range
    f1 = x1 ** 2 + x2 ** 2
    f2 = (x1 - 5) ** 2 + (x2 - 5) ** 2
    f1 = np.reshape(f1, f1.size)
    f2 = np.reshape(f2, f2.size)
    fig.add_trace(go.Scatter(name='Pareto front', x=f1, y=f2, mode='lines', line=dict(color='blue', width=5)))

    if show is True:
        fig.show()
    return fig


def plot_search_results(func, search_results, title='Test', x_label='x', y_label='y', n_samples=50, show=False, save=False, save_options=None):
    # plot a few samples of the objective space and the Pareto front
    fig = func(n_samples=n_samples, show=show)

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
    # fig.add_trace(go.Scatter(x=))


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
    result_ = SearchInstance.search()
    # arr = np.array([i.get_y() for i in result['search_history']])
    # print(arr)
    # print(arr.shape)
    # print(arr[0])
    # print(arr[:, 0])
    return result_


def main():
    return example_1()


def create_plots(articulation_type, benchmark_problem, func, n_samples=50, x_label='f1(x1, x2)', y_label='f2(x1, x2)'):
    # load pickled dictionaries into list of dictionaries
    load_path = '/home/kemal/Programming/Python/Articulation/data/pickles/' + articulation_type + '/' + benchmark_problem + '/'
    os.chdir(load_path)
    file_names = os.listdir()
    dicts = []
    for p in file_names:
        file = open(load_path + p, 'rb')
        dicts.append(pickle.load(file))
        file.close()

    # Some mappings for pretty outputs
    name_mappings = {
        'aposteriori': 'A posteriori',
        'progressive': 'Progressive',
        'apriori': 'A priori'
    }

    save_path = '/home/kemal/Programming/Python/Articulation/data/txts_and_plots/' + articulation_type + '/' + benchmark_problem + '/' + 'plots' + '/'

    # create plots for every dictionary
    for d in dicts:
        search_results = dict(
            search_history=d['search_history'],
            global_best_sol=d['global_best_sol']
        )
        save_options = dict(
            path=save_path,
            name=str(d['test_ID']+'.png')
        )
        plot_search_results(func, search_results, title=str(name_mappings[articulation_type] + ' ' + benchmark_problem),
                            x_label=x_label, y_label=y_label, n_samples=n_samples, show=False, save=True, save_options=save_options)


if __name__ == '__main__':
    create_plots('aposteriori', 'BK1', plot_BK1_objective_space)

