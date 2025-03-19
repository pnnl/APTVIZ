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
import pandas as pd
import scipy
import networkx as nx
import itertools
import matplotlib
import matplotlib.pyplot as plt

def vectorize_row(row, atom_types):
    vec = np.zeros(len(atom_types))
    for i,a in enumerate(atom_types):
        vec[i] = row[a]/100
    return vec

def get_edge_attributes(G, name):
    edges = G.edges(data=True)
    return dict((x[:-1], x[-1][name]) for x in edges if name in x[-1])

def plot_graph(g, color_by, use_cutoff=False, saveto='None', pos='None'):
    # get unique groups
    groups = set(nx.get_node_attributes(g,color_by).values())
    mapping = dict(zip(sorted(groups),itertools.count()))
    
    # create number for each group to allow use of colormap
    nodes = g.nodes()
    colors = [mapping[g.nodes[n][color_by]] for n in nodes]
    
    edge_weights=np.array(list(get_edge_attributes(g, 'similarity').values()))
    edge_widths=1-(edge_weights-edge_weights.min())/(edge_weights.max()-edge_weights.min())
    
    cmap = matplotlib.cm.get_cmap(plt.cm.tab20, len(mapping))
    new_cmap = matplotlib.colors.ListedColormap(cmap.colors[:len(mapping)])

    fig,ax = plt.subplots(figsize=(16,10))
    
    if pos == 'None':
        pos = nx.spring_layout(g, k=0.2)

    nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, 
                                node_color=colors, cmap=new_cmap, 
                                node_size=20, alpha=0.9)
    
    if use_cutoff:
        nx.draw_networkx_edges(g, pos, edgelist=g.edges(), 
                               width=edge_widths, alpha=0.6)
    
    cbar=fig.colorbar(nc, ticks=range(len(mapping)), 
                      spacing='proportional')#, extend='both', extendfrac='auto')
    
    
    cbar.ax.get_yaxis().set_ticks([])
    for j, lab in enumerate(list(mapping.keys())):
        cbar.ax.text(.5, (2 * j + 1) / 8.0, lab, ha='center', va='center', fontsize=12, color='white')
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Connected Component ID', fontsize=14)

    
    plt.axis('off')
    plt.tight_layout()
    if saveto != 'None':
        plt.savefig(save, dpi=150)
    plt.show()

    return pos


def binning(df, atom_types, bins=8):
    # bin by number of atoms
    df['atom_bin']=pd.cut(df['N_atoms'], bins=bins)
    df['voxel_bin']=pd.cut(df['voxel'], bins=bins)
    
    # binning for each atom percent
    for a in atom_types:
        df[f'{a}_bin']=pd.cut(df[a], bins=bins)

    # center of voxels
    for i in ['x','y','z']:
        df[i]=(df[f'bound_{i}_min']+df[f'bound_{i}_max'])/2

    return df


def get_eigenvectors(g):
    # compute unnormalized laplacian L
    L = nx.laplacian_matrix(g, weight='None')

    # compute eigenvectors of L
    evals,evecs=scipy.linalg.eigh(L.todense())

    # sort eigenvectors by eigenvalues
    eigenvectors=np.stack([x for _, x in sorted(zip(evals, evecs))])

    return eigenvectors
