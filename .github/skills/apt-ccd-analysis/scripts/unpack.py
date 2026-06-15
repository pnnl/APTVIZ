import numpy as np
from .OPTICSAPT.APTPosData import APTPosData

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
