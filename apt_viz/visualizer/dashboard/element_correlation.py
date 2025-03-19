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

import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import plotly.graph_objects as go
from dash import dcc,html
import numpy as np


def format_text(text):
    """
    Formats text. used to make the labels for each element in the figure more readable and accurate.
    """
    text=text.replace('1','').replace('2','<sub>2</sub>').replace('3','<sub>3</sub>')
    return f"{text}<sup>+</sup>"


def fetch_correlation(df):
    """
    Creates a correlation plot from a dataframe by calculating the ks statistic
    """
    atom_types = [c[0:] for c in df.columns if (c[0] == 'p' and not 'partition' in c)]
    tmp = pd.DataFrame()
    #CHANGED 'kmeans' TO 'community' FOR NOW. EVENTUALLY CAN PROBABLY GET THE MEAN KS STATISTIC FROM CONFIG FILES.
    for name, group in df.groupby('community')[atom_types]:
        for a in atom_types:
            stats = st.ks_2samp(df[a], group[a])
            # print(stats)
            # print('statistic', stats.statistic)
            sign = 1
            try:
                sign = stats.statistic_sign
            except:
                print('there was no sign in the ks statistic dict')
            finally:
                signed_statistic = sign * stats.statistic
            tmp = pd.concat([tmp, pd.DataFrame({'cluster': [name],
                                                'ion': [format_text(a.replace('p', ''))],
                                                'stat': [signed_statistic]})])
    print("stats recalculated")
    # maybe save the dataframe somewhere, so it can be easily read instead of recalculated

    ks_data = []
    for name, group in reversed(tuple(tmp.groupby('cluster'))):
        ks_data.append(group['stat'].tolist())

    # for name, group in tmp.groupby('cluster'):
    #     ks_data.append(group['stat'].tolist())

    x_ticks = [x + 0.5 for x in range(len(atom_types))]
    x_labels = [format_text(a.replace('p', '')) for a in atom_types]
    #y_labels = [f"Cluster {i}" for i in np.sort(df.kmeans.unique())]

    # CHANGED kmeans TO community HERE
    y_labels = [f"Cluster {int(i)}" for i in sorted(df.community.unique(), reverse=True)]

    data = go.Heatmap(
        z=ks_data,
        x=x_labels,
        y=y_labels,
        #colorscale="RdBu_r",
        #zmid=0,
        hovertemplate='<b>%{z}</b><extra></extra>',
        coloraxis='coloraxis'
    )

    fig = go.Figure(data=data)

    # add invisible trace for keeping data. This allows for showing/hiding clusters without recalculating the whole plot since there is no 'visibility' attribute for each row/cluster to toggle.
    fig.add_trace(data)
    fig["data"][1]["visible"] = False

    fig.update_layout(hovermode='y',
                      coloraxis={'colorbar_len': 1.2,
                                 'colorscale': 'RdBu_r'})

    fig.update_xaxes(side='top', tickangle=-45)

    return fig
