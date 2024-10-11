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

import sys
import os
import os.path as op
import pandas as pd
import glob
import json
from multiprocessing import Pool
from sklearn import cluster
import time

### must have scipy v1.10.1
import scipy.stats as st

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, type=str, help='Location of neighborhood files.')
parser.add_argument('--savedir', required=True, type=str, help='Location to store clustering results.')
parser.add_argument('-r','--r', required=True, type=float, help='Radius of neighborhood.')
parser.add_argument('-o','--overlap', required=True, type=float, help='Neighborhood overlap.')
parser.add_argument('-k','--k', required=True, type=int, help='Number of k-means clusters.')
parser.add_argument('-s','--seeds', required=True, nargs='+', help='List of seeds for k-means clustering.')
args = parser.parse_args()

start = time.time()

# set up directory structure
savefile = f'kmeans_over_all_{args.r}nm-radius_{args.overlap}-overlap_{args.k}clusters'
if not op.isdir(args.savedir):
    os.mkdir(args.savedir)

# check for previously computed files of the same name
if op.isfile(savefile+'.csv') or op.isfile(savefile+'.json'):
    print(f"Clustering already performed in {save_file}; exiting")
    sys.exit()

# read in precomputed neighborhoods
datafile = f'_{args.r}nm-radius_{args.overlap}-overlap.csv'
print(f"Reading precomputed neighborhoods from {op.join(args.datadir, '*'+datafile)}")

df = pd.DataFrame()
print(f"...{len(glob.glob(op.join(args.datadir, '*'+datafile)))} samples found")
for radii in glob.glob(op.join(args.datadir, '*'+datafile)):
    tmp = pd.read_csv(radii)
    tmp['sample']=radii.split('/')[-1].replace(datafile,'')
    tmp = tmp.loc[tmp['outer_layer']==False]
    df = pd.concat([df,tmp], sort=False, ignore_index=True)

# fill with 0 if ion not present in sample
df.fillna(0, inplace=True)

# infer atom types
atom_types = [c for c in df.columns if c[0]=='p']

# extract data to cluster
data=df[atom_types].to_numpy()

print(f"...{data.shape[0]} neighborhoods")
print(f"...{data.shape[1]} ion types: {[a[1:] for a in atom_types]}")

for seed in args.seeds:
    seed = int(seed)
    # run kmeans clustering
    kmeans = cluster.MiniBatchKMeans(n_clusters=args.k, random_state=seed, n_init=10).fit(data)
    df[f"kmeans{seed}"]=kmeans.labels_
    df[f"kmeans{seed}"]=df[f"kmeans{seed}"].astype("category")

    # calculate ks statistics for each cluster
    ks_stats=[]
    for name, group in df.groupby(f"kmeans{seed}")[atom_types]:
        def get_stat(ion, bulk=df, group=group):
            stats=st.ks_2samp(bulk[ion], group[ion])
            return stats.statistic_sign*stats.statistic
        with Pool(len(atom_types)) as p:
            stats=p.map(get_stat, atom_types)
        ks_stats.append(stats)

    # save metadata as .json
    jsonsavefile = op.join(args.savedir, savefile+f'_{seed}seed')
    metadata={'r': args.r, 'k': args.k, 'seed': seed, 'n_neighborhoods': data.shape[0], 'n_iter': kmeans.n_iter_, 'n_steps': kmeans.n_steps_, 'n_features_in': kmeans.n_features_in_, 'inertia': kmeans.inertia_, 'cluster_centers': {i:list(c) for i,c in enumerate(kmeans.cluster_centers_)},  'ions': [a[1:] for a in atom_types], 'ks_statistics': {i:k for i,k in enumerate(ks_stats)}}

    with open(jsonsavefile+".json", "w") as outfile:
        json.dump(metadata, outfile)

# save cluster results
df.to_csv(op.join(args.savedir,savefile+".csv"), index=False)

print(f"Clustering finished in {(time.time()-start)/60:0.2f} min")
