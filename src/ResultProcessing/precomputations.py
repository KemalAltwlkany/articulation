# Script used to precompute values, i.e. the TNK search space is difficult to visualize neatly without many points
import math as math
import numpy as np
import pickle as pickle
import os as os
import copy as copy
import random as random
import plotly.graph_objects as go


def precompute_OSY_space_and_front(n_samples=100, n_samples_Pareto=100, save=True, save_folder=None, save_name=None, show=False):
    if save_folder is None:
        save_folder = '/home/kemal/Programming/Python/Articulation/data/precomputed_data/'
    if save_name is None:
        save_name = 'OSY_data.pickle'

    save_data = dict()

    f1 = []
    f2 = []
    random.seed(0)
    while len(f1) < n_samples*50:  # 5k sample points
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        x3 = random.uniform(1, 5)
        x4 = random.uniform(0, 6)
        x5 = random.uniform(1, 5)
        x6 = random.uniform(0, 10)
        if x1 + x2 - 2 < 0:
            continue
        if 6 - x1 - x2 < 0:
            continue
        if 2 - x2 + x1 < 0:
            continue
        if 2 - x1 + 3 * x2 < 0:
            continue
        if 4 - (x3 - 3)**2 - x4 < 0:
            continue
        if (x5 - 3)**2 + x6 - 4 < 0:
            continue
        f1.append(-25*(x1 - 2)**2 - (x2 - 2)**2 - (x3 - 1)**2 - (x4 - 4)**2 - (x5 - 1)**2)
        f2.append(x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 + x6 ** 2)

    save_data['search_f1'] = copy.deepcopy(f1)
    save_data['search_f2'] = copy.deepcopy(f2)
    fig = go.Figure()
    fig.add_trace(
            go.Scatter(name='Objective space', x=f1, y=f2, mode='markers', marker=dict(color='orange'), marker_size=5))

    f1 = []
    f2 = []

    # The Pareto front can be obtained by going region-by-region:
    # For the entire Pareto front, x4=0, x6=0
    # Work is primarily motivated by:
    # "K. Deb, A. Pratap, T. Meyarivan - Constrained Test Problems for Multi-objective Evolutionary Optimization"

    # region AB
    x1, x2, x5 = 5, 1, 5
    x3_space = np.linspace(1, 5, n_samples_Pareto)
    for x3 in x3_space:
        f1.append(-25 * math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1 ** 2 + x2 ** 2 + x3 ** 2 + x5 ** 2)

    # region BC
    x1, x2, x5 = 5, 1, 1
    x3_space = np.linspace(1, 5, n_samples_Pareto)
    for x3 in x3_space:
        f1.append(-25 * math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1 ** 2 + x2 ** 2 + x3 ** 2 + x5 ** 2)

    # region CD
    x1_space = np.linspace(4.056, 5, n_samples_Pareto)
    x3, x5 = 1, 1
    for x1 in x1_space:
        x2 = (x1 - 2.) / 3.
        f1.append(-25 * math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1 ** 2 + x2 ** 2 + x3 ** 2 + x5 ** 2)

    # region DE
    x1, x2, x5 = 0, 2, 1
    x3_space = np.linspace(1, 3.732, n_samples_Pareto)
    for x3 in x3_space:
        f1.append(-25 * math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1 ** 2 + x2 ** 2 + x3 ** 2 + x5 ** 2)

    # region EF
    x1_space = np.linspace(0, 1, n_samples_Pareto)
    x3, x5 = 1, 1
    for x1 in x1_space:
        x2 = 2 - x1
        f1.append(-25 * math.pow(x1 - 2, 2) - math.pow(x2 - 2, 2) - math.pow(x3 - 1, 2) - 16 - math.pow(x5 - 1, 2))
        f2.append(x1 ** 2 + x2 ** 2 + x3 ** 2 + x5 ** 2)

    # Update 15.08.2020.
    # Construct Pareto set
    n = n_samples_Pareto
    x1 = np.concatenate((5*np.ones(n*2), np.linspace(4.056, 5, n), np.zeros(n), np.linspace(0, 1, n)))
    x2_help = np.linspace(2.056, 3, n)/3
    x2 = np.concatenate((np.ones(n*2), x2_help, 2*np.ones(n), np.linspace(-2, -1, n)))
    x3 = np.concatenate((np.linspace(1, 5, n*2), np.ones(n), np.linspace(1, 3.732, n), np.ones(n)))
    x4 = np.zeros(n*5)
    x5 = np.concatenate((5*np.ones(n), np.ones(n*4)))
    x6 = np.zeros(n*5)


    f1 = np.sum()

    fig.add_trace(
            go.Scatter(name='Pareto front', x=f1, y=f2, mode='markers', marker=dict(color='blue'), marker_size=5))

    save_data['pareto_f1'] = copy.deepcopy(f1)
    save_data['pareto_f2'] = copy.deepcopy(f2)

    if show is True:
        fig.show()
    if save is True:
        os.chdir(save_folder)
        with open(save_name, 'wb') as f:
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)

def precompute_TNK_space_and_front(n_samples=100, n_samples_Pareto=1000, save_folder=None, save_name=None, show=False):
    if save_folder is None:
        save_folder = '/home/kemal/Programming/Python/Articulation/data/precomputed_data/'
    if save_name is None:
        save_name = 'TNK_data.pickle'

    save_data = dict()

    x1_space = np.linspace(0, math.pi, n_samples)  # should be extra dense, because of the non-convex border
    x2_space = np.linspace(0, math.pi, n_samples)
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
    save_data['search_f1'] = copy.deepcopy(f1)
    save_data['search_f2'] = copy.deepcopy(f2)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name='Objective space', x=f1, y=f2, mode='markers', marker=dict(color='orange'), marker_size=5))


    f1 = []
    f2 = []
    x1_space = np.linspace(0, math.pi, n_samples_Pareto)
    x2_space = np.linspace(0, math.pi, n_samples_Pareto)
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

    fig.add_trace(go.Scatter(name='Pareto front', x=f1_pom, y=f2_pom, mode='markers', line=dict(color='blue', width=5)))
    save_data['pareto_f1'] = copy.deepcopy(f1_pom)
    save_data['pareto_f2'] = copy.deepcopy(f2_pom)
    # Update 15.08.2020.
    # Pareto set == Pareto front
    x = np.column_stack((np.array(f1_pom), np.array(f2_pom)))
    save_data['x'] = copy.deepcopy(x)
    os.chdir(save_folder)
    with open(save_name, 'wb') as f:
        pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)

    if show is True:
        fig.show()

def precompute_BK1_objective_space(n_samples=100, n_samples_Pareto=100, save=True, save_folder=None, save_name=None, show=False):
    if save_folder is None:
        save_folder = '/home/kemal/Programming/Python/Articulation/data/precomputed_data/'
    if save_name is None:
        save_name = 'BK1_data.pickle'

    save_data = dict()

    # The objective space - a few samples
    x1_range = np.linspace(-5, 10, n_samples)
    x2_range = np.linspace(-5, 10, n_samples)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    f1 = x1 ** 2 + x2 ** 2
    f2 = (x1 - 5) ** 2 + (x2 - 5) ** 2
    f1 = np.reshape(f1, f1.size)
    f2 = np.reshape(f2, f2.size)

    save_data['search_f1'] = copy.deepcopy(f1)
    save_data['search_f2'] = copy.deepcopy(f2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name='Objective space', x=f1, y=f2, mode='markers', marker=dict(color='orange'), marker_size=5))

    # The Pareto front
    x1_range = np.linspace(0, 5, n_samples_Pareto)
    x2_range = np.linspace(0, 5, n_samples_Pareto)
    #x1, x2 = np.meshgrid(x1_range, x2_range)
    x1, x2 = x1_range, x2_range
    f1 = x1 ** 2 + x2 ** 2
    f2 = (x1 - 5) ** 2 + (x2 - 5) ** 2
    f1 = np.reshape(f1, f1.size)
    f2 = np.reshape(f2, f2.size)
    fig.add_trace(go.Scatter(name='Pareto front', x=f1, y=f2, mode='markers', line=dict(color='blue', width=5)))

    save_data['pareto_f1'] = copy.deepcopy(f1)
    save_data['pareto_f2'] = copy.deepcopy(f2)

    # Update 15.08.2020. - I also need to save the Pareto set values.
    x1 = np.reshape(x1, x1.size)
    x2 = np.reshape(x2, x2.size)
    x = np.column_stack((x1, x2))
    save_data['x'] = copy.deepcopy(x)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(name='Pareto set', x=x[:, 0], y=x[:, 1], mode='markers', marker=dict(color='red'), marker_size=5)
    )
    if save is True:
        os.chdir(save_folder)
        with open(save_name, 'wb') as f:
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)

    if show is True:
        fig.show()
        fig2.show()

def precompute_IM1_objective_space(n_samples=100, n_samples_Pareto=100, save=True, save_folder=None, save_name=None, show=False):
    if save_folder is None:
        save_folder = '/home/kemal/Programming/Python/Articulation/data/precomputed_data/'
    if save_name is None:
        save_name = 'IM1_data.pickle'

    save_data = dict()

    # The objective space - a few samples
    x1_range = np.linspace(1, 4, n_samples)
    x2_range = np.linspace(1, 2, n_samples)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    f1 = 2 * np.sqrt(x1)
    f2 = x1 * (1 - x2) + 5
    f1 = np.reshape(f1, f1.size)
    f2 = np.reshape(f2, f2.size)

    save_data['search_f1'] = copy.deepcopy(f1)
    save_data['search_f2'] = copy.deepcopy(f2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name='Objective space', x=f1, y=f2, mode='markers', marker=dict(color='orange'), marker_size=5))

    # The Pareto front
    x1_range = np.linspace(1, 4, n_samples_Pareto)
    x2_range = 2 * np.ones(n_samples_Pareto)
    # x1, x2 = np.meshgrid(x1_range, x2_range)
    x1, x2 = x1_range, x2_range
    f1 = 2 * np.sqrt(x1)
    f2 = x1 * (1 - x2) + 5
    f1 = np.reshape(f1, f1.size)
    f2 = np.reshape(f2, f2.size)
    fig.add_trace(go.Scatter(name='Pareto front', x=f1, y=f2, mode='lines', line=dict(color='blue', width=5)))

    save_data['pareto_f1'] = copy.deepcopy(f1)
    save_data['pareto_f2'] = copy.deepcopy(f2)

    # Update 15.08.2020. - I also need to save the Pareto set values.
    x1 = np.reshape(x1, x1.size)
    x2 = np.reshape(x2, x2.size)
    x = np.column_stack((x1, x2))
    save_data['x'] = copy.deepcopy(x)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(name='Pareto set', x=x[:, 0], y=x[:, 1], mode='markers', marker=dict(color='red'), marker_size=5)
    )
    if save is True:
        os.chdir(save_folder)
        with open(save_name, 'wb') as f:
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)

    if show is True:
        fig.show()
        fig2.show()

def precompute_SCH1_objective_space(n_samples=100, n_samples_Pareto=100, save=True, save_folder=None, save_name=None, show=False):
    if save_folder is None:
        save_folder = '/home/kemal/Programming/Python/Articulation/data/precomputed_data/'
    if save_name is None:
        save_name = 'SCH1_data.pickle'

    save_data = dict()

    # The objective space - a few samples
    x1 = np.linspace(-7, 7, n_samples * 10)
    f1 = x1 ** 2
    f2 = (x1 - 2) ** 2
    f1 = np.reshape(f1, f1.size)
    f2 = np.reshape(f2, f2.size)

    save_data['search_f1'] = copy.deepcopy(f1)
    save_data['search_f2'] = copy.deepcopy(f2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name='Objective space', x=f1, y=f2, mode='markers', marker=dict(color='orange'), marker_size=5))

    # The Pareto front
    x1 = np.linspace(0, 2, n_samples_Pareto)
    f1 = x1 ** 2
    f2 = (x1 - 2) ** 2
    f1 = np.reshape(f1, f1.size)
    f2 = np.reshape(f2, f2.size)
    fig.add_trace(go.Scatter(name='Pareto front', x=f1, y=f2, mode='lines', line=dict(color='blue', width=5)))

    save_data['pareto_f1'] = copy.deepcopy(f1)
    save_data['pareto_f2'] = copy.deepcopy(f2)

    # Update 15.08.2020. - I also need to save the Pareto set values.
    save_data['x'] = copy.deepcopy(x1)
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(name='Pareto set', x=x1, y=np.zeros(x1.size), mode='markers', marker=dict(color='red'), marker_size=5)
    )

    if save is True:
        os.chdir(save_folder)
        with open(save_name, 'wb') as f:
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)

    if show is True:
        fig.show()
        fig2.show()

def precompute_FON_objective_space(n_samples=10, n_samples_Pareto=10, save=True, save_folder=None, save_name=None, show=False):
    if save_folder is None:
        save_folder = '/home/kemal/Programming/Python/Articulation/data/precomputed_data/'
    if save_name is None:
        save_name = 'FON_data.pickle'

    save_data = dict()

    # The objective space - a few samples
    # It can be computed using 2 dimensions always, as the number of dimensions does not influence the objective space
    n_dims = 2
    x_range = np.array((np.linspace(-4, 4, n_samples),) * n_dims)
    mesh = np.array(np.meshgrid(*x_range))
    a = mesh.reshape((n_dims, n_samples ** n_dims))
    n = n_dims
    sum1 = np.sum(np.square(a - 1. / math.sqrt(n)), axis=0)
    sum2 = np.sum(np.square(a + 1. / math.sqrt(n)), axis=0)
    f1 = 1 - np.exp(-sum1)
    f2 = 1 - np.exp(-sum2)

    save_data['search_f1'] = copy.deepcopy(f1)
    save_data['search_f2'] = copy.deepcopy(f2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name='Objective space', x=f1, y=f2, mode='markers', marker=dict(color='orange'), marker_size=5))

    # The Pareto front
    # Since tests were performed using 5-dimensional alternatives, the code is somewhat hardcoded to create a
    # Pareto-set which corresponds to a 5-dim search space.
    n_dims = 5
    x_range = np.array((np.linspace(-1./math.sqrt(n_dims), 1./math.sqrt(n_dims), n_samples_Pareto),) * n_dims)
    a = np.column_stack((x_range[0, :], x_range[1, :], x_range[2, :], x_range[3, :], x_range[4, :]))
    n = n_dims
    sum1 = np.sum(np.square(a - 1. / math.sqrt(n)), axis=1)
    sum2 = np.sum(np.square(a + 1. / math.sqrt(n)), axis=1)
    f1 = 1 - np.exp(-sum1)
    f2 = 1 - np.exp(-sum2)

    fig.add_trace(go.Scatter(name='Pareto front', x=f1, y=f2, mode='markers', line=dict(color='blue', width=5)))

    save_data['pareto_f1'] = copy.deepcopy(f1)
    save_data['pareto_f2'] = copy.deepcopy(f2)

    save_data['x'] = copy.deepcopy(a)
    if save is True:
        os.chdir(save_folder)
        with open(save_name, 'wb') as f:
            pickle.dump(save_data, f, pickle.HIGHEST_PROTOCOL)

    if show is True:
        fig.show()


if __name__ == '__main__':
    #precompute_BK1_objective_space(n_samples_Pareto=1000, show=True)
    #precompute_IM1_objective_space(n_samples_Pareto=1000, show=True)
    #precompute_SCH1_objective_space(n_samples_Pareto=1000, show=True)
    #precompute_FON_objective_space(n_samples=100, n_samples_Pareto=1000, show=True)
    #precompute_TNK_space_and_front(n_samples=200, n_samples_Pareto=2000, show=True)
    precompute_OSY_space_and_front(show=True)
    #print('Not active.')

