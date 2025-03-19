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

import pandas as pd
import os.path as op
from sklearn import cluster
import json
import scipy.stats as st
from multiprocessing import Pool


def cluster_these_grids(filepath, method, n_clusters, seed):
    """
    Clusters data from a csv file using kmeans, adds a column for the clustering results to the original dataframe and saves as a new csv. returns the name of the csv saved.
    This version is no longer used.
    """
    count_col = 'N_atoms'
    df = pd.read_csv(filepath)
    print("read file")
    df.fillna(0, inplace=True)
    atom_types = [c[0:] for c in df.columns if c[0] == 'p']
    data = df[atom_types].to_numpy()  

    if method == "kmeans":
        kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(data)
        df[method] = kmeans.labels_
        df[method] = df[method].astype("category")

    to_save = op.splitext(filepath)[0] + '_clustered.csv'
    df.to_csv(to_save)
    print("saved clustered file")
    return to_save


def count_clusters(filepath):
    """
    Reads a pre-clustered csv file, and identifies how many clusters are present by looking at the 'kmeans' column. returns the number of clusters.
    This is used if a pre-clustered file is uploaded. should probably add a 'cluster col name' input field in this case. to do later.
    """
    df = pd.read_csv(filepath)
    if 'kmeans' in df.columns:
        n_clusters = max(df['kmeans']) + 1
    else:
        n_clusters = max(df['community']) + 1
    return n_clusters


def get_kvals(filepath):
    """
    Reads a pre-clustered csv file and identifies which k values were tested, if applicable.
    """
    df = pd.read_csv(filepath)

    kvals = []

    if 'kmeans' in df.columns.values:
        print("single k means column detected")
        k = count_clusters(filepath)
        kvals = [k]
    else:
        print("multiple kmeans cols detected")
        cols = df.columns.values.tolist()
        kmeans_cols = [col for col in cols if 'kmeans' in col]
        for col in kmeans_cols:
            temp = col.replace('kmeans', '*')
            temp = temp.replace('_', '*')
            k = int(temp.split('*')[1])
            kvals.append(k)

    return kvals


def get_iters(filepath):
    """
    Reads a pre-clustered csv file and identifies how many iterations were done of kmeans.
    """
    df = pd.read_csv(filepath)

    n_iters = 1
    if 'kmeans' in df.columns.values:
        print("single k means column detected, only one iteration performed")
        return n_iters
    else:
        print('multiple kmeans cols detected, counting number of iterations performed')
        cols = df.columns.values.tolist()
        kmeans_cols = [col for col in cols if 'kmeans' in col]
        for col in kmeans_cols:
            i = col.split('_')[-1]
            if int(i) > int(n_iters):
                n_iters = int(i)

    return n_iters + 1


def multiple_kmeans(filepath, k, n_repeats):
    """
    Runs k means on a sample using a single k value, and multiple different seed values.
    Saves json files for metdata on each run
    Returns a clustered data frame
    """
    df = pd.read_csv(filepath)
    if 'outer_layer' in list(df.columns.values):
        df = df.loc[df['outer_layer']==False]
    ion_types = sorted([c[0:] for c in df.columns if c[0]=='p'])
    data = df[ion_types].to_numpy()

    for seed in range(n_repeats):
        # run kmeans clustering
        kmeans = cluster.MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10).fit(data)
        df[f"kmeans{k}_{seed}"]=kmeans.labels_
        df[f"kmeans{k}_{seed}"]=df[f"kmeans{k}_{seed}"].astype("category")

        # calculate ks statistics for each cluster
        ks_stats=[]
        for name, group in df.groupby(f"kmeans{k}_{seed}")[ion_types]:
            
            def get_stat(ion, bulk=df, group=group):
                stats=st.ks_2samp(bulk[ion], group[ion])
                return stats.statistic_sign*stats.statistic
            
            ion_stat = []
            for ion in ion_types:
                stats = get_stat(ion)
                ion_stat.append(stats)
            print(f"len of ion stat: {len(ion_stat)}")
            ks_stats.append(ion_stat)
        print(f"len of ks_stats: {len(ks_stats)}")
        # save metadata as .json
        json_to_save = op.splitext(filepath)[0] + f'_{k}k_{seed}seed.json'
        metadata={'r': 1.0, 
                'k': k, 
                'seed': seed, 
                'n_neighborhoods': data.shape[0], 
                'n_iter': kmeans.n_iter_, 
                'n_steps': kmeans.n_steps_, 
                'n_features_in': kmeans.n_features_in_, 
                'inertia': kmeans.inertia_, 
                'cluster_centers': {i:list(c) for i,c in enumerate(kmeans.cluster_centers_)},  
                'ions': [a[1:] for a in ion_types], 
                'ks_statistics': {i:k for i,k in enumerate(ks_stats)}}

        with open(json_to_save, "w") as outfile:
            json.dump(metadata, outfile)

    return df


if __name__ == "__main__":
    cluster_these_grids('../../data/2nm_gb_grid_radii.csv', 5, 42)
