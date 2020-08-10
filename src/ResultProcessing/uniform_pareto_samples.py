import numpy as np
import math as math
import plotly.graph_objects as go
from src.PreferenceArticulation.BenchmarkObjectives import MOO_Problem


def FON_samples(n_dims=2, n_samples=5):
    fun = MOO_Problem.FON
    x1 = np.linspace(-1. / math.sqrt(n_dims), 1. / math.sqrt(n_dims), n_samples)
    x2 = x1[::1]  # return copy of x1 but reversed
    f1, f2 = [], []
    for i in range(n_samples):
        x, y = fun(np.array(x1[i], x2[i]))
        f1.append(x)
        f2.append(y)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(name='Objective space', x=f1, y=f2, mode='markers', marker=dict(color='orange'), marker_size=5))
    #fig.show()
    print(x1)
    print(x2)
    print(f1)
    print(f2)


if __name__ == '__main__':
    FON_samples(n_dims=2)

