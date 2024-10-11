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

import os
import os.path as op
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
from sklearn.metrics.cluster import adjusted_mutual_info_score, adjusted_rand_score
import numpy as np
import json
import seaborn as sns
from scipy.spatial import distance
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, type=str, help='Location of APT neighborhood files.')
parser.add_argument('-r','--radius', default=1.0, type=float, help='Radius of neighborhood.')
parser.add_argument('-o','--overlap', default=0.5, type=float, help='Neighborhood overlap.')
parser.add_argument('-k','--k', nargs='+', help='List of k-means clusters.')
parser.add_argument('-s','--seeds', nargs='+', help='List of seeds for k-means clustering.', required=True)
args = parser.parse_args()

start = time.time()

palette_color = sns.color_palette('tab20')

savedir=op.join(args.datadir, 'processed')

if not op.isdir(savedir):
    os.mkdir(savedir)
    
sim_scores_file = op.join(savedir,f'kmeans_cluster_similarity_{args.radius}nm-radius_{args.overlap}-overlap.csv')
sim_graph_json = op.join(savedir,f'kmeans_cluster_graph_{args.radius}nm-radius_{args.overlap}-overlap.json')
sim_fig_file = op.join(savedir, f"kmeans_cluster_pie_{args.radius}nm-radius_{args.overlap}")

cluster_descriptor = 'ks_statistics'


sim_scores=pd.DataFrame()
G=nx.Graph()
n=0
c=0
for n_clusters in args.k:

    file = op.join(args.datadir, f'kmeans_over_all_{args.radius}nm-radius_{args.overlap}-overlap_{n_clusters}clusters.csv')

    if op.isfile(file):
        print(f'Reading clusters from {file}')
        cf = pd.read_csv(file, usecols=[f"kmeans{s}" for s in args.seeds], dtype='uint8') 
    else:
        print(f'Missing clustering file -- skipping {file}')
        break

    if n==0:
        with open(file) as f:
            line = f.readline()
        ions=[i[1:] for i in line.split(',') if i[0]=='p']

    # pie plot of cluster counts
    fig,ax=plt.subplots(1,len(args.seeds), figsize=(14,2))
    for i, seed in enumerate(args.seeds):
        vc=cf[f"kmeans{seed}"].value_counts()
        ax[i].set_title(f"k-clusters={n_clusters}\nseed={seed}")
        ax[i].pie(vc.values, colors=palette_color)
    plt.tight_layout()
    plt.savefig(sim_fig_file+f"-{n_clusters}means.png", dpi=300)
    plt.close()

    # AMI and RAND scores
    combos = list(itertools.combinations(args.seeds, r=2))

    def get_ami(combo, cf=cf):
        return adjusted_mutual_info_score(cf[f"kmeans{combo[0]}"], cf[f"kmeans{combo[1]}"])
    def get_rand(combo, cf=cf):
        return adjusted_rand_score(cf[f"kmeans{combo[0]}"], cf[f"kmeans{combo[1]}"])

    with Pool(20) as p:
        ami=p.map(get_ami, combos)
        rand=p.map(get_rand, combos)

    combos = np.array(combos)
    tmp = pd.DataFrame({'cluster_a': combos.T[0],'cluster_b': combos.T[1], 'AMI':ami, 'RAND':rand})
    tmp['seed']=seed
    tmp['kmeans']=n_clusters
    sim_scores=pd.concat([sim_scores,tmp], sort=False, ignore_index=True)

    # add nodes to graph based on cluster
    for seed in args.seeds:
        with open(file.replace('.csv',f'_{seed}seed.json')) as f:
            data=json.load(f)

        # set embedding in order of ions
        for cluster in data[cluster_descriptor]:
            cluster_ks=dict(zip(data['ions'],data[cluster_descriptor][cluster]))
            embedding_ks=[]
            for ion in ions:
                if ion in cluster_ks.keys():
                    embedding_ks.append(cluster_ks[ion])
                else:
                    embedding_ks.append(0)

            # add node for each cluster to graph
            # attrs: ks, seed, cluster, radius, overlap, count
            G.add_node(n, ks=embedding_ks, seed=seed, cluster=int(cluster), k_means=n_clusters,
                       N_mil=len(cf.loc[cf[f'kmeans{seed}']==int(cluster)])/1000000,
                       )
            for i,ion in enumerate(ions):
                nx.set_node_attributes(G, {n:round(embedding_ks[i],2)}, ion)
            n+=1
        c+=1

    # add edges based on euclidean distance between ks stats
    ks_embeddings=nx.get_node_attributes(G, 'ks')
    G.add_weighted_edges_from([(e[0][0],e[0][1],e[1]) for e in zip(itertools.combinations(range(len(G)), 2),
                                                                   distance.pdist(list(ks_embeddings.values()), 'cosine'))])
    weights=np.array(list(nx.get_edge_attributes(G,'weight').values()))

    # clear memory
    cf=pd.DataFrame()
    data=[]


# save sim_scores
sim_scores.to_csv(sim_scores_file, index=False)

# save graph
json_object = json.dumps(G, default=nx.node_link_data)
# Writing to sample.json
with open(sim_graph_json, "w") as outfile:
    outfile.write(json_object)

print(f"KS statistics collection finished in {(time.time()-start)/60:0.2f} min")
