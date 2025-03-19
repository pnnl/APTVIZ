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

import numpy as np
import plotly.graph_objects as go
from dashboard.colors import get_colorscale, get_color_from_index
import pandas as pd

def fetch_agreement(df):
    #TODO: CHANGE ERROR BAR FROM STD TO 95% CONFIDENCE INTERVAL
    cols = [col for col in df.columns if 'partition' in col]

    partition_df = df[cols].copy()
    partition_voters = sorted(pd.unique(partition_df[cols].values.ravel('K')))

    mean_agreements = []
    community_lst = []
    stdev_agreements = []
    community_names = []
    for community in sorted(df['community'].unique()):
        community_lst.append(community)
        community_names.append(f"{int(community)}")
        agrees = df.loc[df['community'] == community, 'agreement'].tolist()
        mean_agreements.append(np.mean(agrees)*100)
        stdev_agreements.append(np.std(agrees)*100)
        # for each community, get the mean and stdev for agreements. the mean goes in the list which is the y. stdev will be errorbar

    colorscale = get_colorscale(len(partition_voters))
    # print('agreement: ', colorscale)

    fig = go.Figure(
        go.Bar(x=community_names, y=mean_agreements,
               error_y={
                   'array':stdev_agreements,
                   'thickness': 1,
                   'color': '#1D2951'
                   },
               marker={
                   'color':community_lst,
                   'colorscale':colorscale
               })
    )

    fig.update_traces(marker_line_color='#1D2951', marker_line_width=1)
    fig.update_layout(width=500, height=500,
        xaxis={
            'title':'Community',
            'titlefont_size': 20,
            'titlefont_color': '#1D2951',
            'tickfont_size':15,
            'tickfont_color': '#1D2951'
        },
        yaxis={
            'title': "Agreement (%)",
            'titlefont_size': 20,
            'titlefont_color': '#1D2951',
            'tickfont_size': 15,
            'tickfont_color': '#1D2951',
        })
    return fig


def fetch_disagreement(df):
    cols = [col for col in df.columns if 'partition' in col]
    # print('cols: ', cols)
    partition_df = df[cols].copy()
    partition_df['community'] = df['community'].tolist()

    # print(df['community'].unique())
    partition_voters = sorted(pd.unique(partition_df[cols].values.ravel('K')))
    
    disagreeing_percent_df = pd.DataFrame()
    disagreeing_percent_df['Disagreeing Community'] = [int(community) for community in partition_voters]
    col_names = []
    for community in sorted(df['community'].unique()):
        community_df = partition_df.loc[partition_df['community']==community]
        community_df.drop(columns=['community'])
        counts = np.unique(community_df.values, return_counts=True)
        # print(f"counts0: {counts[0]}, counts1: {counts[1]}")
        
        votes = [0]*len(partition_voters)
        # print(votes)
        for i,voter in enumerate(counts[0]):
            votes[int(voter)] = counts[1][i]

        votes[int(community)] = 0
        # print(f"votes: {votes}")
        percents = [100*val/sum(votes) for val in votes]
        # print("disagreeing communities: ", disagreeing_percent_df['Disagreeing Community'].tolist())
        # print("percents: ", percents)
        col_name = f'{int(community)}'
        col_names.append(col_name)
        disagreeing_percent_df[col_name] = percents
    
    colorscale = get_colorscale(len(partition_voters))
    # print(f"colorscale: {colorscale}")
    color_lst = get_color_from_index(len(partition_voters), 0)
    color_lst = [f"rgb{color}" for color in color_lst]
    # print(color_lst)
    
    fig = go.Figure()
    for voter in partition_voters:
        fig.add_trace(go.Bar(
            x=disagreeing_percent_df.columns[1:],
            y= disagreeing_percent_df.loc[disagreeing_percent_df['Disagreeing Community'] == voter, col_names].values.flatten().tolist(),
            # y=list(disagreeing_percent_df.loc[disagreeing_percent_df['Disagreeing Community']==community][list(disagreeing_percent_df.columns[1:])].transpose().iloc[:,0]),
            name=f"Partition {int(voter)}",
            type='bar',
            marker={
                   'color':color_lst[int(voter)]
               }
        ))
    fig.update_layout(barmode="stack", width=500, height=500,
        xaxis={
            'title':'Community',
            'titlefont_size': 20,
            'titlefont_color': '#1D2951',
            'tickfont_size':15,
            'tickfont_color': '#1D2951'
        },
        yaxis={
            'title': "Percent of Disagreeing Voters",
            'titlefont_size': 20,
            'titlefont_color': '#1D2951',
            'tickfont_size': 15,
            'tickfont_color': '#1D2951',
            
        })
    return fig
