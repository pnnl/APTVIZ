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

import dash
from dash import html, dcc, callback, Input, Output, State, ctx
import os
import json
from dashboard.placeholders import blank_fig
from dashboard.data import create_dataframe
from dashboard.similarity_graph import fetch_similarity_graph_go
from dashboard.agreement import fetch_agreement, fetch_disagreement


dash.register_page(__name__)

##########################
# Layout
##########################

layout = html.Div(
    [
        html.Button('Update Graphs', id='update-button-e', className='big-button', n_clicks=0),

        html.Div(id='dashboard-e', children=[
            html.Div(id='similaritygraph-div', className='fig-div', children=[
                html.H1("K-Cluster Similarity Graph"),
                html.Div(children=[
                    dcc.Loading(dcc.Graph(id='similaritygraph',
                                            figure=blank_fig(),
                                            style={'display':'inline-block'}
                                            ), type='cube')
                ])
            ]),
            html.Div(id='voteagreement-div', className='fig-div', children=[
                html.H1("Community Assignment Agreement"),
                html.Div(children=[
                    dcc.Loading(dcc.Graph(id='voteagreement',
                                            figure=blank_fig(),
                                            style={'display':'inline-block'}
                                            ), type='cube')
                ])
            ]),
            html.Div(id='disagreeingvotes-div', className='fig-div', children=[
                html.H1("Community Assignment Disagreement"),
                html.Div(children=[
                    dcc.Loading(dcc.Graph(id='disagreeingvotes',
                                            figure=blank_fig(),
                                            style={'display':'inline-block'}
                                            ), type='cube')
                ])
            ]),
        ]),
    ])


##########################
# Callbacks
##########################

# updates the similarity graph
@callback(
    Output('similaritygraph', 'figure'),
    Input('update-button-e', 'n_clicks'),
    State('data-store', 'data'),
    # prevent_initial_call=True
)
def update_similarity_graph(n_clicks, data):
    """
    If the 'update' button is clicked, or the data store is updated, the similarity graph figure is updated accordingly.
    In the future, store this figure so it doesn't need to be recreated every time the user switches to the evaluate tab
    """
    triggered_id = ctx.triggered_id
    if triggered_id == 'update-button-e' or n_clicks == 1:
        return create_similarity_graph(data)
    return blank_fig()

def create_similarity_graph(data):
    """
    When the 'update' button is clicked, the similarity graph plot is (re)created by reading in the clustered csv as a dataframe.
    """
    # this hardcoded csv path is no longer used and can be removed after testing
    print(data['clustered_filepath'])
    if not os.path.exists(data['clustered_filepath']):
        print("file does not exist")
        return blank_fig()
    elif 'G' not in data.keys():
        print('one or more arguments required to create the similarity graph were missing. check for G, pos and partition in config file')
        return blank_fig()
    else:
        print('generating similarity graph')
        df = create_dataframe(data['clustered_filepath'])
        fig = fetch_similarity_graph_go(df, data['G'], data['pos'], data['partition'])
    return fig


# updates the agreement chart
@callback(
    Output('voteagreement', 'figure'),
    Input('update-button-e', 'n_clicks'),
    State('data-store', 'data'),
    # prevent_initial_call=True
)
def update_voteagreement(n_clicks, data):
    """
    If the 'update' button is clicked, or the data store is updated, the vote agreement figure is updated accordingly.
    In the future, store this figure so it doesn't need to be recreated every time the user switches to the evaluate tab
    """
    triggered_id = ctx.triggered_id
    if triggered_id == 'update-button-e' or n_clicks == 1:
        return create_voteagreement(data)
    return blank_fig()

def create_voteagreement(data):
    """
    When the 'update' button is clicked, the vote agreement plot is (re)created by reading in the clustered csv as a dataframe.
    """
    # this hardcoded csv path is no longer used and can be removed after testing
    print(data['clustered_filepath'])
    if not os.path.exists(data['clustered_filepath']):
        print("file does not exist")
        return blank_fig()
    else:
        print('generating vote agreement graph')
        df = create_dataframe(data['clustered_filepath'])
        fig = fetch_agreement(df)
    return fig


# updates the disagreeing votes figure
@callback(
    Output('disagreeingvotes', 'figure'),
    Input('update-button-e', 'n_clicks'),
    State('data-store', 'data'),
    # prevent_initial_call=True
)
def update_disagreeingvotes(n_clicks, data):
    """
    If the 'update' button is clicked, or the data store is updated, the disagreeing votes figure is updated accordingly.
    In the future, store this figure so it doesn't need to be recreated every time the user switches to the evaluate tab
    """
    triggered_id = ctx.triggered_id
    if triggered_id == 'update-button-e' or n_clicks == 1:
        return create_disagreeingvotes(data)
    return blank_fig()

def create_disagreeingvotes(data):
    """
    When the 'update' button is clicked, the disagreeing votes plot is (re)created by reading in the clustered csv as a dataframe.
    """
    # this hardcoded csv path is no longer used and can be removed after testing
    print(data['clustered_filepath'])
    if not os.path.exists(data['clustered_filepath']):
        print("file does not exist")
        return blank_fig()
    else:
        print('generating vote agreement graph')
        df = create_dataframe(data['clustered_filepath'])
        fig = fetch_disagreement(df)
    return fig
