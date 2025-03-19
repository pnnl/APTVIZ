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
import networkx as nx
import ase
from utils import element



def count_rows(file):
    # count number of rows in gzip file
    with gzip.open(file, 'rb') as f:
        for i, l in enumerate(f):
            pass
    return i+1

def remove_duplicate_edges(df):
    # remove duplicate edges between vertices of same atomic_number
    df['pair']=df.apply(lambda x: sorted([x['v1_ID'],x['v2_ID']]), axis=1)
    df['p1']=df['pair'].apply(lambda x: x[0])
    df['p2']=df['pair'].apply(lambda x: x[1])
    df.drop_duplicates(subset=['p1','p2','v1_type','v2_type'], inplace=True)
    # drop columns used to delete duplicated edges
    df.drop(['pair','p1','p2'], axis=1, inplace=True)
    return  df



def remove_noise(positions, ids, key='Noise'):
    # delete noise ions
    noise_idx = np.where(np.array(ids)==key)
    xyzcoords = np.delete(positions, noise_idx, 0)
    xyzids = np.delete(ids, noise_idx, 0)
    
    return xyzcoords, xyzids

        
def write_xyz(xyzcoords, xyzids, savefile='coords.xyz', noise=False):
    ## write coords to .xyz file
    if noise:
        xyzcoords, xyzids = remove_noise(xyzids, xyzcoords)
    
    # convert to angstrom
    xyzcoords = xyzcoords*10
    
    # format in .xyz style
    x=xyzcoords.T[0]
    y=xyzcoords.T[1]
    z=xyzcoords.T[2]
    lines=[f'{len(x)}\n','\n']+['   '.join(x)+'\n' for x in np.array(list(zip(xyzids,x,y,z)))]

    # write to file
    with open(savefile, 'w') as f:
        f.writelines(lines)
        
def midpoint(p):
    p1=p[0]
    p2=p[1]
    return p1 + (p2-p1)/2   


########## GRAPH UTILS ###########

def atoms_to_graph(atoms, k=1):
    # get distance matrix
    dm = atoms.get_all_distances(mic=True)

    # get van der Waals radii of all atoms
    atomic_numbers=atoms.get_atomic_numbers()
    vdw = [element.vdw_radii[x] for x in atomic_numbers]

    # perform outer addition to get pairwise vdw sums
    vdw_sums = np.add.outer(vdw,vdw)

    # multiplier k
    vdw_sums *= k

    # generate mask by subtracting vdw_sums from distances
    # and setting negatives to 0 and positives to 1
    # and set diagonal to 0
    mask = np.where(vdw_sums - dm <0, 0, 1)
    np.fill_diagonal(mask,0)

    # get adjacency matrix
    A = dm * mask

    # convert adjacency matrix to graph
    G = nx.from_numpy_matrix(A)
    
    # set node attributes
    nx.set_node_attributes(G, dict(zip(range(len(atomic_numbers)), atomic_numbers)), name='atomic_number')
    
    # set edge attributes
    nx.set_edge_attributes(G, k, name='kNN')

    return G

def generate_graph(atoms, k_max=5):
    """
    Input:
        atoms (ase.Atoms): atoms object
        k_max (int): maximum kNN for edge generation
    Returns:
        G (nx.Graph): graph with edges up to kNN
    """
    
    kNN_graphs=[atoms_to_graph(atoms, k=k) for k in range(1, k_max+1)]
    
    G = kNN_graphs[0].copy()
    for N in kNN_graphs[1:]:
        # remove edges that already exist in graph
        D = nx.difference(N, G)

        # add new edges to S
        # where attributes conflict, uses attributes of S (i.e., keeps lowest kNN)
        G = nx.compose(N,G)
        
    return G



