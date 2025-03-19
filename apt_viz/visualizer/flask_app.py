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

from dash import Dash
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from flask import Flask
from werkzeug.serving import run_simple
from dash_home_layout import dashhomelayout
from dash import Output, Input


#####################################
# Name Servers
#####################################

app = Flask(__name__)

dash_app = Dash(
    __name__,
    use_pages=True,
    server = app,
    url_base_pathname='/dashboard/',
    external_stylesheets=[
            '/static/dist/css/styles.css',
            '/static/dist/css/dash_styles.css',
            'https://fonts.googleapis.com/css?family=Lato',
        ]
    )
dash_app.layout = dashhomelayout()

#####################################
# Dashboard Home Page Tab Navigation
#####################################

@dash_app.callback(
    Output('tab-1', 'className'),
    [Input('tab-1', 'className'),
    Input('location', 'pathname')]
)
def update_visualize_page(className, pathname):
    if pathname == '/dashboard/visualize-dashboard':
        new_class = str(className) + ' active'
        print("visualize active")
        return new_class
    else:
        new_class = str(className).replace(' active', '')
        print("visualize not active")
        return new_class
    

@dash_app.callback(
    Output('tab-2', 'className'),
    [Input('tab-2', 'className'),
    Input('location', 'pathname')]
)
def update_evaluate_page(className, pathname):
    if pathname == '/dashboard/evaluate-dashboard':
        new_class = str(className) + ' active'
        print("evaluate active")
        return new_class
    else:
        new_class = str(className).replace(' active', '')
        print("evaluate not active")
        return new_class

#####################################
# Flask Routes
#####################################

with app.app_context():
        from utils import routes



#####################################
# Werkzeug Serving
#####################################

served_app = DispatcherMiddleware(app, {
    'dashboard': dash_app.server
})

#####################################
# Run Flask App
#####################################

run_simple('0.0.0.0', 8080, served_app, use_reloader=True, use_debugger=True)
