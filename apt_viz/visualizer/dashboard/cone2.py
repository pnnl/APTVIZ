'''
This material was prepared as an account of work sponsored by an agency of the
United States Government.  Neither the United States Government nor the United
States Department of Energy, nor Battelle, nor any of their employees, nor any
jurisdiction or organization that has cooperated in the development of these
materials, makes any warranty, express or implied, or assumes any legal
liability or responsibility for the accuracy, completeness, or usefulness or
any information, apparatus, product, software, or process disclosed, or
represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by
trade name, trademark, manufacturer, or otherwise does not necessarily
constitute or imply its endorsement, recommendation, or favoring by the United
States Government or any agency thereof, or Battelle Memorial Institute. The
views and opinions of authors expressed herein do not necessarily state or
reflect those of the United States Government or any agency thereof.

PACIFIC NORTHWEST NATIONAL LABORATORY
operated by
BATTELLE
for the
UNITED STATES DEPARTMENT OF ENERGY
under Contract DE-AC05-76RL01830
'''

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go


def fetch_cone(df, clusters):
    """
    Creates a 3d cone, colored by cluster.
    Inputs: df, a dataframe created from the clustered csv file
            clusters, a list of selected clusters to display
    """

    cmap = {i: matplotlib.colors.to_hex(c) for i, c in enumerate(plt.cm.tab20.colors)}

    df.reset_index(inplace=True)
    df['outer_layer'] = False
    for name, group in df.groupby('midpoint_z'):
        df.loc[df['index'].isin(group.groupby('midpoint_x')['midpoint_y'].idxmax().tolist()), 'outer_layer'] = True
        df.loc[df['index'].isin(group.groupby('midpoint_y')['midpoint_x'].idxmax().tolist()), 'outer_layer'] = True
        df.loc[df['index'].isin(group.groupby('midpoint_x')['midpoint_y'].idxmin().tolist()), 'outer_layer'] = True
        df.loc[df['index'].isin(group.groupby('midpoint_y')['midpoint_x'].idxmin().tolist()), 'outer_layer'] = True

    # cluster_choices = [0, 1, 2, 3, 4]
    subset = df.loc[(df['community'].isin(clusters)) & (df['outer_layer'] == False)]

    cone_fig = go.Figure(
        data=[
            go.Scatter3d(
                x=subset['midpoint_x'],
                y=subset['midpoint_y'],
                z=subset['midpoint_z'],
                opacity=1,
                hoverinfo='skip',
                mode='markers',
                marker=dict(color=subset['community'].apply(lambda x: cmap[x]), size=5)
            )
        ]
    )

    camera = dict(eye=dict(x=2, y=2, z=0.1))
    cone_fig.update_layout(scene_camera=camera)
    cone_fig.update_layout(scene_aspectmode='manual',
                           scene_aspectratio=dict(x=1, y=1, z=2))

    cone_fig.update_layout(scene=dict(
        xaxis=dict(
            showticklabels=False,
            backgroundcolor="rgb(250, 212, 248)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white", ),
        yaxis=dict(
            showticklabels=False,
            backgroundcolor="rgb(212, 241, 250)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white"),
        zaxis=dict(
            showticklabels=False,
            backgroundcolor="rgb(255, 244, 207)",
            gridcolor="white",
            showbackground=True,
            zerolinecolor="white", ), ),
        #width=700,
        margin=dict(
            r=10, l=10,
            b=10, t=50),
        xaxis_showspikes=False,
        yaxis_showspikes=False
    )

    x_min = df['midpoint_x'].min()
    x_max = df['midpoint_x'].max()
    y_min = df['midpoint_y'].min()
    y_max = df['midpoint_y'].max()
    z_min = df['midpoint_z'].min()
    z_max = df['midpoint_z'].max()

    z_val = z_min
    buffer = abs(x_max - x_min) * 0.2

    cone_fig.add_trace(go.Mesh3d(
        x=[x_min - buffer, x_min - buffer, x_max + buffer, x_max + buffer],
        y=[y_min - buffer, y_max + buffer, y_min - buffer, y_max + buffer],
        z=[z_val] * 4,
        color='rgb(0,0,0)',
        #alphahull=0,
        delaunayaxis='z',
        opacity=0.75,
        hoverinfo='skip'
    ))
    return cone_fig
