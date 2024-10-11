'''
                         DISCLAIMER
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

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from networkx.readwrite import json_graph
import json
import matplotlib
import pandas as pd
import os.path as op

def read_json_file(filename):
    """Read in networkx graph from json file
    
    Args:
        filename (str): Path to json file.
    """
    with open(filename) as f:
        js_graph = json.load(f)

    ions = list(js_graph['nodes'][0].keys())
    for c in ['ks','seed','cluster','k_means','N_mil','id']:
        ions.remove(c)
    return json_graph.node_link_graph(js_graph), ions

def format_text(text):
    if text[-1]=='1':
        text = text[:-1]+'^{{+}}'
    if text[-1]=='2':
        text = text[:-1]+'^{2{+}}'
    text=text.replace('1','').replace('2','_{2}').replace('3','^{3}').replace('4','^{4}')
    return "$\mathrm{"+text+"}$"

def plot_partitions(G, pos, partition, edge_weights, saveas=''):
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
    if len(saveas)>0:
        plt.savefig(saveas, dpi=300)
    plt.show()
    

def write_xyz(df, savefile, tag='partition'):
    coords = ['   '.join([str(i) for i in list(c)]+['\n']) for c in np.array(df[[tag,'midpoint_x','midpoint_y','midpoint_z']])]
    with open(savefile, "w") as f:
        f.writelines([str(len(coords))+'\n', " \n"])
        f.writelines(coords)
