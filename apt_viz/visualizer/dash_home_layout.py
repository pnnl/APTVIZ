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
from dash import dash_table, dcc, html, Output, Input, State, ctx, ALL
import os
import json

root_path = os.getcwd()
if os.path.exists(os.path.join(root_path, 'config.json')):
    with open(os.path.join(root_path, 'config.json'), 'r') as read_file:
        config = json.load(read_file)
else:
    print("file not found")
    config = {}


def dashhomelayout():
    return html.Div(children=[
        dcc.Store(id="data-store", data=config),
        # dcc.Location(refresh="callback-nav", id='location'),
        dcc.Location(refresh=True, id='location'),
        html.Div(className='tabs', children=[
                html.Div(
                    dcc.Link("Visualize", href=dash.page_registry['pages.visualize_dashboard']['relative_path'], className='link', id='tab-1'), className='tab'
                ),
                html.Div(
                    dcc.Link("Evaluate", href=dash.page_registry['pages.evaluate_dashboard']['relative_path'], className='link', id='tab-2'), className='tab'
                ),
                html.Div(
                    className='tabsShadow'
                ),
                html.Div(
                    className='glider'
                )
            ]),
        # dash.page_container
    html.Div(dash.page_container, id='content-window')
    ])
