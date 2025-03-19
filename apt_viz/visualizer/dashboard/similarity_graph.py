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
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mc
import math
from dashboard.colors import get_colorscale


def fetch_similarity_graph(df, G, pos, partition):
    """
    This instance is not used. It is here as a reference for how this plot was originally generated in matplotlib.
    """

    # convert the dictionary of dictionary format back to graphs
    G = nx.Graph(G)

    edge_weights=np.array(list(nx.get_edge_attributes(G,'weight').values()))
    # color the nodes according to their partition
    node_colors=list(partition.values())
    n_colors = len(set(node_colors))
    cmap = matplotlib.colors.ListedColormap(plt.cm.tab20.colors[:n_colors])

    fig,ax = plt.subplots(1,2,figsize=(8,5), gridspec_kw={'width_ratios':[5,1]})

    # edge width based on weight
    edge_width = (edge_weights-edge_weights.min())/(edge_weights.max()-edge_weights.min())


    # draw nodes
    nodes = nx.draw_networkx_nodes(G, pos, nodelist=G.nodes(), node_color=node_colors, edgecolors='black',
                                    node_size=50, cmap=cmap, ax=ax[0])

    # draw edges
    arcs=nx.draw_networkx_edges(G, pos, edgelist=G.edges(), width=edge_width, ax=ax[0])


    # set colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_colors))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax[1])
    cbar.ax.tick_params(labelsize=11)
    cbar.ax.set_yticks([x+0.5 for x in range(n_colors)],range(1,1+n_colors))
    cbar.set_label(label='Community', size=12)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.tight_layout()
    print('figure created')

    return fig


def fetch_similarity_graph_go(df, G, pos, partition):
    """
    This is the version that is used to generate the plotly graph objects version of the figure.
    """
    # convert the dictionary of dictionary format back to graphs
    G = nx.Graph(G)

    #NOTE: pos is a dictionary of x, y, positions keyed by node.

    edgelist=G.edges()
    nodelist=G.nodes()
    # print(f"nodes: {nodelist}")
    # print(f"pos dict: {pos}")
    # print(f"edges: {edgelist}")

    # create edges
    edge_x = []
    edge_y = []
    for edge in edgelist:
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    # color the nodes according to their partition
    node_colors=list(partition.values())
    n_colors = len(set(node_colors))
    print(n_colors)
    color_indices = list(range(n_colors))
    
    colorscale = get_colorscale(n_colors)

    color_names = []
    for index in color_indices:
        color_names.append(f"{index}")
    all_tick_indices = np.linspace(0, n_colors-1, n_colors*2 + 1)
    tick_indices = [all_tick_indices[i] for i in range(1, len(all_tick_indices), 2)]

    print('similarity graph: ', colorscale)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            # showscale=True,
            # reversescale=True,
            size=10,
            colorbar=dict(
                orientation='h', 
                thickness=15,
                title='Partition',
                titleside='top',
                yanchor='bottom',
                y=-0.25,
                tickvals=tick_indices,
                ticktext=color_names,
                tickfont={
                    'color':'#1D2951',
                    'size': 15
                }
                
            ),
            colorscale=colorscale,
            color=node_colors,
            line_width=2))
    
    

    # color node points by number of connections (TODO: change this to cluster)
    # node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        # node_adjacencies.append(len(adjacencies[1]))
        node_text.append('# of connections: '+str(len(adjacencies[1])))

    # node_trace.marker.color = node_adjacencies
    # node_trace.marker.color = color_list
    node_trace.text = node_text

    # create figure
    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                width = 500,
                height = 500,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                )


    return fig
