# THIS FILE IS NOT USED.
# DIDN'T DELETE IT BECAUSE SOME CODE FRAGMENTS MIGHT BE USEFUL IN THE FUTURE.


import numpy as np
import matplotlib.pyplot as plt
from PreferenceArticulation.BenchmarkObjectives import *


def bi_objective_scatter_plot(f1, f2, x, title="sample", xlabel=None, ylabel=None, ):
    fig = plt.figure()
    f1_values = []
    f2_values = []
    for xi in x:
        f1_values.append(f1(xi))
        f2_values.append(f2(xi))
    plt.scatter(f1_values, f2_values)
    plt.show()


def create_linspace_vector(x1_start, x1_stop, x2_start, x2_stop, samples):
    """
    Unlike create_mesh_vector, both x1 and x2 are now sampled independently and then the corresponding indices are
    just grouped together.
    :param x1_start: start coordinate of x1 variable
    :param x1_stop: stop coordinate of x1 variable
    :param x2_start: start coordinate of x2 variable
    :param x2_stop: stop coordinate of x2 variable
    :param samples: number of samples to be created between start and stop coordinate of both variables.
    :return: Function returns a numpy vector of size samples * 1, where every element is a numpy vector of size
        1 * 2.

    Example usage

    """
    return np.linspace((x1_start, x2_start), (x1_stop, x2_stop), samples)


def create_mesh_vector(x1_start, x1_stop, x2_start, x2_stop, samples):
    """
    Function is used to create a vector containing pairs (x1, x2) of 2D data so that it can be evaluated and plotted.
    Effectively, this is just a vector containing the same data as the two mesh vectors given by numpy itself.

    :param x1_start: specifies starting coordinate of x1 variable
    :param x1_stop:  stop coordinate of x1 variable
    :param x2_start: start coordinate of x2 variable
    :param x2_stop: stop coordinate of x2 variable
    :param samples: number of samples to be evenly spaced between the start and stop coordinates of both variables.
            Equal for both variables, in order to create a mesh.
    :return: returns a np array/vector of size samples**2 x 1. Every element is a 1x2 numpy array, where the first
        element is the x1 coordinate and the second element the x2 coordinate.

    Example of usage:
    x = create_mesh_vector(-4, 4, -2, 2, 4)
    returns:
        [[-4.         -2.        ]
        [-1.33333333 -2.        ]
        [ 1.33333333 -2.        ]
        [ 4.         -2.        ]
        [-4.         -0.66666667]
        [-1.33333333 -0.66666667]
        [ 1.33333333 -0.66666667]
        [ 4.         -0.66666667]
        [-4.          0.66666667]
        [-1.33333333  0.66666667]
        [ 1.33333333  0.66666667]
        [ 4.          0.66666667]
        [-4.          2.        ]
        [-1.33333333  2.        ]
        [ 1.33333333  2.        ]
        [ 4.          2.        ]]
    """
    x1, x2 = np.meshgrid(np.linspace(x1_start, x1_stop, samples), np.linspace(x2_start, x2_stop, samples))
    x1, x2 = x1.flatten(), x2.flatten()
    x1, x2 = np.reshape(x1, (samples**2, 1)), np.reshape(x2, (samples**2, 1))
    x1 = np.concatenate((x1, x2), axis=1)
    return x1


def example1():
    """
    Basically plots the envelope/borderline of a meshgrid. In order to plot, uses linspace combinations.
    :return:
    """
    x = np.linspace((0, 0), (10, 10), 1000)
    bi_objective_scatter_plot(Objectives_2D.f1, Objectives_2D.f2, x)


def example2():
    """
    Plots the entire meshgrid, of x,y. A few extra operations have been added to make this more compatible with my
    BenchmarkObjectives class.
    :return:
    """
    x, y = np.meshgrid(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
    x, y = x.flatten(), y.flatten()
    x, y = np.reshape(x, (100**2, 1)), np.reshape(y, (100**2, 1))
    x = np.concatenate((x, y), axis=1)
    bi_objective_scatter_plot(Objectives_2D.f1, Objectives_2D.f2, x)


def plot_2_objectives():
    pass


if __name__ == '__main__':
    # example1()
    # example2()
    print(create_mesh_vector(-4, 4, -2, 2, 4))

