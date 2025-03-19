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
from dash import html, dcc, callback, Input, Output, dash_table, State, ctx
from flask import current_app as app
import plotly.graph_objects as go
import os
import json
from dashboard.data import create_dataframe, collect_z_slice, create_marks
from dashboard.animation import fetch_animation
from dashboard.element_correlation import fetch_correlation
from dashboard.cone2 import fetch_cone
from dashboard.layout import html_layout
from dashboard.dynamic_styles import checklist_options, checklist_values
from dashboard.placeholders import blank_fig
import flask
from dashboard.dash_layout import get_layout
import dash_daq as daq


dash.register_page(__name__)
print('current working directory', os.getcwd())
root_path = os.getcwd()

###############################
# layout
###############################

if os.path.exists(os.path.join(root_path, 'config.json')):
    with open(os.path.join(root_path, 'config.json'), 'r') as read_file:
        config = json.load(read_file)
    filepath = config['filepath']
    n_clusters = int(config['n_clusters'])
else:
    print("file not found")
    config = {}
    n_clusters = 1

layout = html.Div(
    [
        # dcc.Store(id="data-store", data=config),
        html.Button('Update Graphs', id='update-button', className='big-button', n_clicks=0),
        # this store will eventually be used to save the filename. have to figure out how to get it here though
        dcc.Store(id='session', storage_type='session'),

        html.Div(id='dashboard', children=[
            # column that contains all visualizations
            html.Div(id='fig-container', children=[

                #row 1
                html.Div(id='row-1', children=[

                    # z slice animation
                    html.Div(id='animation-div', className='fig-div', children=[
                        html.H1('Z Slice Animation'),
                        dcc.Loading(dcc.Graph(id='z_slice_animation',
                                                figure=blank_fig(),
                                                ), type='cube')
                    ]),

                    # z cone
                    html.Div(id='cone-div', className='fig-div', children=[
                        html.H1('3D Representation'),
                        dcc.Loading(dcc.Graph(id='3d_cone',
                                                figure=blank_fig()
                                                ), type='cube'),
                        html.Div(id='z_slice_text')
                    ]),
                # end of row 1
                ]),

                #row 2
                html.Div(id='row-2', children=[

                    #element correlation
                    html.Div(id='correlation-div', className='fig-div', children=[
                        html.H1("Element Correlations Between Clusters"),
                        html.Div(children=[
                            dcc.Loading(dcc.Graph(id='ele_corr',
                                                    figure=blank_fig(),
                                                    style={'height':300}
                                                    ), type='cube')
                        ])
                    ]),
                # end of row 2
                ]),

            # end of fig container
            ]),

            # column that contains the user inputs
            html.Div(id='menu-container', className='fig-div', children=[

                html.H1("Options"),

                html.Div(id='cluster-picker', children=[
                    html.H3("Show Specific Communities:"),
                    dcc.Checklist(options=checklist_options([0]),
                                value=checklist_values(0),
                                id='cluster-list')
                ]),
                # html.Div(children=[
                #     html.H3("Select Cluster Colors:"),
                #     html.Div(children=[
                #         html.Div(children=[
                #             daq.ColorPicker(
                #                 id='my-color-picker-1',
                #                 size=164,
                #                 #label='Color Picker',
                #                 value=dict(hex='#119DFF')
                #             ),
                #             html.Div(id='color-picker-output-1')
                            
                #         ])
                #     ])  
                # ], style={'margin-left': '10px'}),

                html.Div(id='scale-picker', children=[
                    html.H3("Change Color Bar Range:"),
                    dcc.Slider(min=0, max=1, step=0.1, value=1,
                                marks=None,
                                tooltip={"placement": 'bottom', 'always_visible': True},
                                id='cb_scale')
                ]),
            # end of menu-container
            ]),
        # end of dashboard
        ]),
    # end of layout
    ], style={"margin": 10})


###############################
# callbacks
###############################

@callback(
    Output('data-store', 'data'),
    Input('update-button', 'n_clicks'),
    # prevent_initial_call=True
)
def refresh_data(n_clicks):
    """
    This is triggered when a user clicks the 'update' button. It is used to regenerate all figures. As the figures currently do not load when the page is first displayed, this
    must be called for the figures to generate for the first time as well. Eventually, there will be an overlay on the page, with a button that also triggers this callback for a more
    professional appearance. This function reads from the config file and returns the dictionary as 'data'.
    A mapping is also created from cluster name to cluster index (in case some clusters have been omitted in community detection)
    """
    with open(os.path.join(app.root_path, 'config.json'), 'r') as read_file:
        config = json.load(read_file)
    data = config

    # df = create_dataframe(data['clustered_filepath'])
    # clusters = sorted(df['community'].unique())
    # cluster_mapping = {}
    # for i, cluster in enumerate(clusters):
    #     cluster_mapping[cluster: i]
    # print("cluster mapping: (cluster to index) ", cluster_mapping)
    # data['cluster_mapping'] = cluster_mapping
    # data['communities'] = clusters
    return data

@callback(
    Output('cluster-list', 'options'),
    Input('update-button', 'n_clicks'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def refresh_checklist_options(n_clicks, data):
    """
    This is triggered by the 'update' button, and uses data stored in the 'data-store' to check how many clusters are present in the data. It returns a list of checklist items,
    each with a label and a value corresponding to one of the clusters.
    """
    df = create_dataframe(data['clustered_filepath'])
    clusters = clusters = sorted(df['community'].unique())
    options = checklist_options(clusters)
    # print('from refresh checklist options', options)
    return options

@callback(
    Output('cluster-list', 'value'),
    Input('cluster-list', 'options'),
    prevent_initial_call=True
)
def refresh_checklist_values(available_options):
    """
    This is triggered by a change in the checklist options. It ensures that all checklist options are checked when a refresh occurs (?)
    """
    value = [cluster['value'] for cluster in available_options]
    # print("from refresh checklist values", value)
    return value

@callback(
    Output('3d_cone', 'figure'),
    Input('update-button', 'n_clicks'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def create_cone(n_clicks, data):
    """
    This is triggered by the 'update' button, and uses data in the 'data-store' to create the 3d cone figure.
    """
    # fp = os.path.join(app.root_path, 'data', '2nm_gb_grid_radii_clustered.csv')
    if not os.path.exists(data['clustered_filepath']):
        print("from create cone: file does not exist")
        return blank_fig()
    else:
        df = create_dataframe(data['clustered_filepath'])
    clusters = [cluster for cluster in range(data['n_clusters'])]
    # print("from create_cone", clusters)
    fig = fetch_cone(df, clusters)
    return fig

# updates the z animation when cluster list is changed or update button is pressed
# TODO: this is not robust. must create z animation before updating it!
@callback(
    Output('z_slice_animation', 'figure'),
    Input('cluster-list', 'value'),
    Input('update-button', 'n_clicks'),
    State('z_slice_animation', 'figure'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def update_z_animation(selected_clusters, n_clicks, fig, data):
    """
    If the 'update' button is clicked, the 2d animation figure will be created from scratch.
    If the cluster list is changed, the 2d animation figure will be updated to show only the selected clusters.
    """
    triggered_id = ctx.triggered_id
    if triggered_id == 'update-button' or n_clicks == 1:
        return create_z_animation(data['clustered_filepath'])
    elif triggered_id == 'cluster-list':
        print("from update z animation", selected_clusters)
        return update_animation_clusters(selected_clusters, fig)
    return

def create_z_animation(filepath):
    """
    Creates the 2d animation figure using data in the 'data store'. The dataframe is built from the clustered csv file, then the animation is built.
    """
    # this hard-coded filepath is no longer used. can be deleted once testing complete
    if not os.path.exists(filepath):
        print("from create z animation: file does not exist")
        return blank_fig()
    else:
        df = create_dataframe(filepath)
    fig = fetch_animation(df)
    return fig

def update_animation_clusters(selected_clusters, fig):
    """
    Updates the 2d animation figure to show the clusters selected in the cluster checklist.
    """
    for i, fig_data in enumerate(fig["data"]):
        if i in selected_clusters:
            fig_data["visible"] = True
        else:
            fig_data["visible"] = "legendonly"

    return fig

# updates a text indicator of what z slice we're looking at
# @dash_app.callback(
#     Output(component_id='z_slice_text', component_property='children'),
#     Input('ele_corr', 'figure')
# )
# def update_output_div(fig):
#     print(fig)
#     #print(fig['frames'])
#     return f'Z slice: {"z"}'

# updates the element correlation based on selected clusters and color scale value
@callback(
    Output('ele_corr', 'figure'),
    Input('cluster-list', 'value'),
    Input('cb_scale', 'value'),
    Input('update-button', 'n_clicks'),
    State('ele_corr', 'figure'),
    State('data-store', 'data'),
    prevent_initial_call=True
)
def update_element_correlation(selected_clusters, cb_scale, n_clicks, fig, data):
    """
    If the 'update' button is clicked, the list of selected clusters changes, or the color scale value is updated, the element correlation figure is updated accordingly.
    """
    triggered_id = ctx.triggered_id
    if triggered_id == 'update-button' or n_clicks == 1:
        return create_element_correlation(data['clustered_filepath'])
    elif triggered_id == 'cluster-list' or triggered_id == 'cb_scale':
        return update_correlation_clusters(selected_clusters, cb_scale, fig, data)
    return

def create_element_correlation(filepath):
    """
    When the 'update' button is clicked, the element correlation plot is (re)created by reading in the clustered csv as a dataframe.
    """
    # this hardcoded csv path is no longer used and can be removed after testing
    fp = os.path.join(app.root_path, 'data', '2nm_gb_grid_radii_clustered.csv')
    if not os.path.exists(filepath):
        print("from create ele corr: file does not exist")
        return blank_fig()
    else:
        df = create_dataframe(filepath)
    fig = fetch_correlation(df)
    return fig

def update_correlation_clusters(selected_clusters, cb_scale, fig, data):
    """
    When the list of selected clusters or the color scale is changed, the correlation figure is updated to match.
    """
    all_z = fig["data"][1]["z"]

    # map from cluster to index 
    # currently makes a new mapping every time because not saving cluster mapping to data store
    if 'cluster_mapping' in data.keys():
        # print('using existing cluster mapping for ks plot')
        selected_cluster_indices = [data['cluster_mapping'][cluster] for cluster in selected_clusters]
    else:
        # print('creating new cluster mapping for ks plot')
        df = create_dataframe(data['clustered_filepath'])
        clusters = sorted(df['community'].unique())
        cluster_mapping = {}
        for i, cluster in enumerate(clusters):
            cluster_mapping[cluster] = i
        # print("cluster mapping: (cluster to index) ", cluster_mapping)
        
        selected_cluster_indices = [cluster_mapping[cluster] for cluster in selected_clusters]
        
    # print(f"selected clusters: {selected_clusters}")
    # print(f"sorted: {sorted(selected_clusters)}")
    # print(f"selected indices: {selected_cluster_indices}")
    selected_z = [all_z[i] for i in sorted(selected_cluster_indices)]
    selected_y = [f"Cluster {i}" for i in sorted(selected_clusters, reverse=True)]
    #for i in selected_clusters:
        #selected_z.append(all_z[i])

    fig["data"][0]["z"] = selected_z
    fig["data"][0]["y"] = selected_y

    if cb_scale <=0.01:
        cmax = 0.01
    else:
        cmax = cb_scale

    cmin = -1 * cmax

    fig = go.Figure(fig)
    fig.update_layout(coloraxis={'cmax': cmax,
                                    'cmin': cmin,
                                    'colorbar_labelalias': {cmax: f'{cmax}, Enriched', '−'+str(cmax): f'−{cmax}, Depleted'}})

    return fig

# @callback(
# Output('color-picker-output-1', 'children'),
# Input('my-color-picker-1', 'value')
# )
# def update_color_output(value):
#     return f'The selected color is {value}.'
