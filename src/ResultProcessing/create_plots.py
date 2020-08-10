import os as os
import pickle as pickle
import plotly.graph_objects as go
from itertools import product
from src.PreferenceArticulation.BenchmarkObjectives import *



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

def plot_IM1_objective_space(n_samples=100, show=True):
    # The objective space - a few samples
    x1_range = np.linspace(1, 4, n_samples)
    x2_range = np.linspace(1, 2, n_samples)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    f1 = 2 * np.sqrt(x1)
    f2 = x1 * (1 - x2) + 5
    f1 = np.reshape(f1, f1.size)
    f2 = np.reshape(f2, f2.size)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name='Objective space', x=f1, y=f2, mode='markers', marker=dict(color='orange'), marker_size=5))

    # The Pareto front
    x1_range = np.linspace(1, 4, n_samples // 10)
    x2_range = 2 * np.ones(n_samples//10)
    # x1, x2 = np.meshgrid(x1_range, x2_range)
    x1, x2 = x1_range, x2_range
    f1 = 2 * np.sqrt(x1)
    f2 = x1 * (1 - x2) + 5
    f1 = np.reshape(f1, f1.size)
    f2 = np.reshape(f2, f2.size)
    fig.add_trace(go.Scatter(name='Pareto front', x=f1, y=f2, mode='lines', line=dict(color='blue', width=5)))

    if show is True:
        fig.show()
    return fig

def plot_SCH1_objective_space(n_samples=100, show=True):
    # The objective space - a few samples
    x1 = np.linspace(-7, 7, n_samples*10)
    f1 = x1**2
    f2 = (x1-2)**2
    f1 = np.reshape(f1, f1.size)
    f2 = np.reshape(f2, f2.size)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name='Objective space', x=f1, y=f2, mode='markers', marker=dict(color='orange'), marker_size=5))

    # The Pareto front
    x1 = np.linspace(0, 2, n_samples//5)
    f1 = x1**2
    f2 = (x1 - 2) ** 2
    f1 = np.reshape(f1, f1.size)
    f2 = np.reshape(f2, f2.size)
    fig.add_trace(go.Scatter(name='Pareto front', x=f1, y=f2, mode='lines', line=dict(color='blue', width=5)))

    if show is True:
        fig.show()
    return fig

def plot_FON_objective_space(n_samples=200, show=True, n_dims=2):
    # It can be computed using 2 dimensions always, as the number of dimensions does not influence the objective space

    x_range = np.array((np.linspace(-4, 4, n_samples), )*n_dims)
    mesh = np.array(np.meshgrid(*x_range))
    a = mesh.reshape((n_dims, n_samples**n_dims))
    n = n_dims
    sum1 = np.sum(np.square(a - 1. / math.sqrt(n)), axis=0)
    sum2 = np.sum(np.square(a + 1. / math.sqrt(n)), axis=0)
    f1 = 1 - np.exp(-sum1)
    f2 = 1 - np.exp(-sum2)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name='Objective space', x=f1, y=f2, mode='markers', marker=dict(color='orange'), marker_size=5))

    # The Pareto front
    x1_range = np.linspace(-1./math.sqrt(n_dims), 1./math.sqrt(n_dims), n_samples // 10)
    x2_range = x1_range.copy()
    x1, x2 = x1_range, x2_range
    f1 = np.square(x1 - 1./math.sqrt(n)) + np.square(x2 - 1./math.sqrt(n))
    f2 = np.square(x1 + 1./math.sqrt(n)) + np.square(x2 + 1./math.sqrt(n))
    f1 = 1 - np.exp(-f1)
    f2 = 1 - np.exp(-f2)
    fig.add_trace(go.Scatter(name='Pareto front', x=f1, y=f2, mode='lines', line=dict(color='blue', width=5)))
    if show is True:
        fig.show()
    return fig

# nsamples is unused, it will be removed from all benchmark functions soon
def plot_TNK_objective_space(n_samples=100, show=True):
    path = '/home/kemal/Programming/Python/Articulation/data/precomputed_data/'
    os.chdir(path)

    file = open(path + 'TNK_data.pickle', 'rb')
    data = pickle.load(file)
    file.close()


    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name='Objective space', x=data['search_f1'], y=data['search_f2'], mode='markers',
                   marker=dict(color='orange'), marker_size=5))
    fig.add_trace(
        go.Scatter(name='Pareto front', x=data['pareto_f1'], y=data['pareto_f2'], mode='markers', marker=dict(color='blue'),
                   marker_size=5))
    if show is True:
        fig.show()
    return fig


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
        save_options = dict(
            path=save_path,
            name=str(d['test_ID'] + extension)
        )
        plot_search_results(articulation_type, benchmark_problem, search_results, title=str(name_mappings[articulation_type] + ' ' + benchmark_problem),
                            x_label=x_label, y_label=y_label, show=False, save=True,
                            save_options=save_options)


if __name__ == '__main__':
    # A POSTERIORI TESTS
    # create_plots('aposteriori', 'BK1')
    # create_plots('aposteriori', 'IM1')
    # create_plots('aposteriori', 'SCH1')
    # create_plots('aposteriori', 'FON')
    # create_plots('aposteriori', 'TNK')
    # create_plots('aposteriori', 'OSY')

    # A PRIORI TESTS
    #create_plots('apriori', 'BK1')
    #create_plots('apriori', 'IM1')
    create_plots('apriori', 'SCH1')


