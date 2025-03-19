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
from dash import dash_table, dcc, html, Output, Input, State, ctx
import dash_daq as daq
from .data import create_dataframe, collect_z_slice, create_marks
from .animation import fetch_animation
from .element_correlation import fetch_correlation
from .cone2 import fetch_cone
from .layout import html_layout
from .dynamic_styles import checklist_options, checklist_values
from .placeholders import blank_fig
import os
import json


def get_layout(root_path):
    """
    Creates the layout of the dashboard page.
    """

    layout = html.Div(
        [
            dcc.Store(id="data-store", data=config),
            html.Button('Update Graphs', id='update-button', className='big-button', n_clicks=0),

            # this store will eventually be used to save the filename. have to figure out how to get it here though
            dcc.Store(id='session', storage_type='session'),


            html.Div(id='dashboard', children=[
                # contains everything in the main row

                # column #1
                # z slice animation
                html.Div(id='animation-div', className='fig-div', children=[
                    html.H1('Z Slice Animation'),
                    dcc.Loading(dcc.Graph(id='z_slice_animation',
                                          figure=blank_fig(),
                                          style={
                                              "width": 700,
                                              "height": 650
                                          }), type='cube')
                ]),

                # column #2
                html.Div(id='col-2', children=[

                    # row #1
                    html.Div(id='row-1', children=[

                        # z cone
                        html.Div(id='cone-div', className='fig-div', children=[
                            html.H1('3D Representation'),
                            dcc.Loading(dcc.Graph(id='3d_cone',
                                                  figure=blank_fig(),
                                                  style={"width": 200,
                                                         "height": 200,
                                                         "margin": "auto"
                                                         }), type='cube'),
                            html.Div(id='z_slice_text')
                        ]),

                        # options
                        html.Div(id='options-div', className='fig-div', children=[
                            #html.H1("Options"),
                            html.Div(children=[
                                html.Div(children=[
                                    html.Div(children=[
                                        html.H3("Show Specific Clusters:"),
                                        # dcc.Checklist(options=checklist_options(create_dataframe(filepath)),
                                        #               value=np.sort(create_dataframe(filepath).kmeans.unique()),
                                        #               id='cluster_list')
                                        dcc.Checklist(options=checklist_options(n_clusters),
                                                  value=checklist_values(n_clusters),
                                                  id='cluster-list')
                                    ]),
                                    html.Div(children=[
                                        html.H3("Select Cluster Colors:"),
                                        html.Div(children=[
                                            html.Div(children=[
                                                daq.ColorPicker(
                                                    id='my-color-picker-1',
                                                    size=164,
                                                    #label='Color Picker',
                                                    value=dict(hex='#119DFF')
                                                ),
                                                html.Div(id='color-picker-output-1')
                                                
                                            ])
                                        ])  
                                    ], style={'margin-left': '10px'}),
                                ], style={'display': 'flex', 'flex-direction': 'row'}),
                                html.Div(children=[
                                    html.H3("Change Color Bar Range:"),
                                    dcc.Slider(min=0, max=1, step=0.1, value=1,
                                               marks=None,
                                               tooltip={"placement": 'bottom', 'always_visible': True},
                                               id='cb_scale')
                                ])

                            ]),

                        ]),

                    ], style={'display': 'flex', 'flex-direction': 'row'}),

                    # row #2: element correlation
                    html.Div(id='correlation-div', className='fig-div', children=[
                        html.H1("Element Correlations in Clusters"),
                        html.Div(children=[
                            dcc.Loading(dcc.Graph(id='ele_corr',
                                                  figure=blank_fig(),
                                                  style={'height': 300}
                                                  ), type='cube')
                        ])
                    ]),

                ], style={'display': 'flex', 'flex-direction': 'column'}),

            ], style={'display': 'flex', 'flex-direction': 'row'}),

        ], style={"margin": 10}

    )
    return layout
