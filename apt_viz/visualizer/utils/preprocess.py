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
import os
import os.path as op
from scipy import spatial
from utils import unpack
from collections import Counter
import csv 
import time

def process_raw(args):
    """
    Taken (and slightly modified) from run_neighborhood_generation.py by Jenna Pope
    Inputs: args is a dictionary with the following entries:
                'pos', type=str, help='Location of APT neighborhood file.'
                'csv', type=str, help='filepath to save csv file to'
                'rrng', type=str, help='Range file.')
                'radius', default=1, type=float, help='Radius of neighborhood.'
                'overlap', default=0.5, type=float, help='Percent overlap of neighborhoods (0-1].'
    """
    start = time.time()


    # flatten list of list
    def flatten(l):
        return [item for sublist in l for item in sublist]



    save_file=args['csv']
    radius = args['radius']
    overlap = args['overlap']

    coords, xyzids, atom_types = unpack.extract_APT_data(op.join(args['pos']), args['rrng'], simplify=False)

    print(f"{len(coords)} ions")

    # KDTree for neighborhood collection
    tree = spatial.cKDTree(coords)

    # extract coordinates
    X,Y,Z = coords.T

    # calculate grid spacing
    step = (2*radius)*(1-overlap)

    # create meshgrid
    xx, yy, zz = np.meshgrid(np.arange(X.min()+radius, X.max()-radius+(step/2), step), 
                            np.arange(Y.min()+radius, Y.max()-radius+(step/2), step), 
                            np.arange(Z.min()+radius, Z.max()-radius+(step/2), step), sparse=False)

    mesh = np.array(list(zip(xx.flatten(), yy.flatten(), zz.flatten())))
    mesh_tree = spatial.cKDTree(mesh)

    # match mesh points to ion tree
    matches=mesh_tree.query_ball_tree(tree, r=radius)

    # collect only non-empty voxels
    nonempty_matches=[(idx, mtch) for idx, mtch in enumerate(matches) if len(mtch)>0]

    field_names=atom_types+['p'+a for a in atom_types]+['d'+a for a in atom_types]+['N_atoms','density','midpoint_x','midpoint_y','midpoint_z','radius','id']

    with open(save_file, 'w') as csv_file:
        dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
        dict_object.writerow({i:i for i in field_names})


    def get_ions(match, savefile, xyzids=xyzids, mesh=mesh, radius=radius, field_names=field_names):
        # get atom info in dictionary form
        ions = xyzids[match[1]]

        # get ion counts
        info = Counter(ions)

        # get percentages 
        n_atoms = len(xyzids[match[1]])
        for key in ions:
            info['p'+key]=info[key]/n_atoms
            info['d'+key]=info[key]/((4/3)*np.pi*radius**3)

        info['N_atoms']=n_atoms
        info['density']=n_atoms/((4/3)*np.pi*radius**3)
        
        # get central coordinates
        pts = mesh[match[0]].astype(np.float32)
        info['midpoint_x']=pts[0]
        info['midpoint_y']=pts[1]
        info['midpoint_z']=pts[2]

        # get voxel info
        info['radius']=radius
        info['id']=match[0]

        for k in list(set(field_names)-set(info.keys())):
            info[k]=0

        info = dict(info)

        with open(savefile, 'a') as csv_file:
            dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
            dict_object.writerow(info)
        return 

    for e in nonempty_matches:
        get_ions(e,save_file)
        
    # denote ions on outer edges
    df = pd.read_csv(save_file)
    df.reset_index(inplace=True)
    df['outer_layer']=False
    for name,group in df.groupby('midpoint_z'):
        df.loc[df['index'].isin(group.groupby('midpoint_x')['midpoint_y'].idxmax().tolist()),'outer_layer']=True
        df.loc[df['index'].isin(group.groupby('midpoint_y')['midpoint_x'].idxmax().tolist()),'outer_layer']=True
        df.loc[df['index'].isin(group.groupby('midpoint_x')['midpoint_y'].idxmin().tolist()),'outer_layer']=True
        df.loc[df['index'].isin(group.groupby('midpoint_y')['midpoint_x'].idxmin().tolist()),'outer_layer']=True
        
    # denote second layer 
    df['second_layer']=False
    for name,group in df.loc[df['outer_layer']==False].groupby('midpoint_z'):
        df.loc[df['index'].isin(group.groupby('midpoint_x')['midpoint_y'].idxmax().tolist()),'second_layer']=True
        df.loc[df['index'].isin(group.groupby('midpoint_y')['midpoint_x'].idxmax().tolist()),'second_layer']=True
        df.loc[df['index'].isin(group.groupby('midpoint_x')['midpoint_y'].idxmin().tolist()),'second_layer']=True
        df.loc[df['index'].isin(group.groupby('midpoint_y')['midpoint_x'].idxmin().tolist()),'second_layer']=True
        
    df.to_csv(save_file, index=False)

    return
