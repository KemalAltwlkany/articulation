import os as os
import pickle as pickle
import copy as copy
import pandas as pd
import numpy as np
import plotly.graph_objects as go

save_path = '/home/kemal/Programming/Python/Articulation/data/handmade_plots/'
# this script is only used to produce the plots which I needed for my thesis

def plot_regions():
    global save_path
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=[[0, 20, 35, 45, 45],
           [20, 20, 35, 45, 45],
           [65, 65, 100, 100, 100],
           [85, 85, 100, 100, 100],
           [85, 85, 100, 100, 100]],
        x=[10, 30, 50, 70, 90],
        y=[10, 30, 50, 70, 90],
        colorscale=[(0.00, "lime"), (0.15, "lime"), (0.15, "lightseagreen"), (0.30, "lightseagreen"), (0.30, "orchid"), (0.45, "orchid"),
                    (0.45, "purple"), (0.60, "purple"), (0.60, "yellow"), (0.75, "yellow"), (0.75, "darkorange"), (0.90, "darkorange"), (0.90, "red"), (1.0, "red")],
        colorbar=dict(
            title='<b>Regions</b>',
            titleside='top',
            tickmode='array',
            tickvals=[7.5, 22.5, 37.5, 52.5, 67.5, 82.5, 95],
            ticktext=['<b>(U)</b>', '<b>(A)</b>', '<b>I1<</b>', '<b>I1<**</b>', '<b>I2<</b>', '<b>I2<**</b>', '<b>I12</b>']
        )
    )
    )

    fig.update_traces(showscale=True)
    fig.update_layout(
        title=dict(
            text='<b>Progressive articulation regions:</b>',
            y=0.93,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(
            title='<b>f1</b>',
            tickmode='array',
            tickvals=[0, 20, 40, 60, 80],
            ticktext=['<b>A<sub>1</sub></b>', '<b>B<sub>1</sub></b>', '<b>C<sub>1</sub></b>', '<b>D<sub>1</sub></b>', '<b>E<sub>1</sub></b>'],

        ),
        yaxis=dict(
            title='<b>f2</b>',
            tickmode='array',
            tickvals=[0, 20, 40, 60, 80],
            ticktext=['<b>A<sub>2</sub></b>', '<b>B<sub>2</sub></b>', '<b>C<sub>2</sub></b>', '<b>D<sub>2</sub></b>', '<b>E<sub>2</sub></b>'],
        )
    )
    #fig.update_traces(showlegend=False)
    #fig.update(layout_showlegend=False)
    fig.show()
    fig.write_image(save_path + 'progressive_regions.png')


def plot_obtained_pareto_front(art_type=None, problem_name=None):
    art_names = dict(
        progressive='Progressive',
        apriori='A priori',
        aposteriori='A posteriori'
    )

    global save_path
    load_path = '/home/kemal/Programming/Python/Articulation/data/pickles/' + art_type + '/' + problem_name + '/'
    entries = os.listdir(load_path)
    f1 = np.zeros(len(entries))
    f2 = np.zeros(len(entries))
    for i in range(len(entries)):
        file = open(load_path + entries[i], 'rb')
        data = pickle.load(file)
        file.close()
        f1[i] = data['global_best_sol'].get_y()[0]
        f2[i] = data['global_best_sol'].get_y()[1]

    fig = go.Figure()
    fig.add_trace(go.Histogram2d(
        name='Obtained Pareto front',
        x=f1,
        y=f2,
        autobinx=False,
        autobiny=False,
        xbins=dict(start=-300, end=0, size=15),
        ybins=dict(start=0, end=200, size=15),
        colorscale='YlGnBu',
        colorbar=dict(
            title='<b>Number of optimums:</b>',
            titleside='top'
        )
    ))
    load_path = '/home/kemal/Programming/Python/Articulation/data/precomputed_data/'
    file = open(load_path + problem_name + '_data.pickle', 'rb')
    data = pickle.load(file)
    file.close()
    f1 = data['pareto_f1']
    f2 = data['pareto_f2']
    fig.add_trace(go.Scatter(name='Actual Pareto front', x=f1, y=f2, mode='markers', line=dict(color='red'), marker_size=7))
    fig.update_layout(title={
        'text': str('<b>' + art_names[art_type] + ' ' + problem_name + ': actual and obtained Pareto front' + '</b>'),
        'y': 0.93,
        'x': 0.5,
        'xanchor': 'center',
        'yanchor': 'top'},
        xaxis_title='<b>f1</b>', yaxis_title='<b>f2</b>', legend_title='<b>Legend:</b>')

    fig.show()
    fig.write_image(save_path + art_type + '_' + problem_name + '_front_density.png')



if __name__ == '__main__':
    #plot_regions()
    plot_obtained_pareto_front('apriori', 'OSY')
    # os.chdir('/home/kemal/Programming/Python/Articulation/data/precomputed_data/')
    # file = open('BK1_data.pickle', 'rb')
    # data = pickle.load(file)
    # file.close()
    # print(data.keys())
    # print(len(data['pareto_f1']))

