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
import itertools

import sys
sys.path.insert(0,'./OPTICS-APT')
from utils.OPTICSAPT.APTPosData import APTPosData


def remove_noise(positions, ids):
    # delete noise ions
    noise_idx = np.where(np.array(ids)=='Noise')
    xyzcoords = np.delete(positions, noise_idx, 0)
    xyzids = np.delete(ids, noise_idx, 0)

    return xyzcoords, xyzids


def extract_APT_data(data_filename, rng_filename, simplify=False):
    PosData = APTPosData()
    PosData.load_data(data_filename, rng_filename)

    # extract element IDs
    ids = PosData.identity

    if simplify:
        atom_types=list(set([x[:-1] for x in list(set(ids)) if x !='Noise']))
        # remove number from after element name to get vdw radii
        ids = np.array(list(map(lambda st: str.replace(st, '1', ''), ids)))
        ids = np.array(list(map(lambda st: str.replace(st, '2', ''), ids)))
    else:
        atom_types=list(set([x for x in list(set(ids)) if x !='Noise']))

    # extract positions
    positions = PosData.pos
    coords = positions[:, :3]

    # remove noise
    coords, xyzids = remove_noise(coords, ids)

    return coords, xyzids, atom_types


def mean_norm(df, col):
    # mean norm of a column
    col_mean = df[col].mean()
    col_std = df[col].std()
    df[f'{col}_norm']=(df[col]-col_mean)/col_std
    return


def raw_data(data_filename, rng_filename, simplify=False):
    # unpack raw data
    coords, xyzids, atom_types = extract_APT_data(data_filename, rng_filename, simplify=simplify)
    
    # put in dataframe
    df = pd.DataFrame({'ion':xyzids,'x':coords.T[0],'y':coords.T[1],'z':coords.T[2]})

    # normalize coords
    for d in ['x','y','z']:
        mean_norm(df, d)
        
    return df, atom_types

