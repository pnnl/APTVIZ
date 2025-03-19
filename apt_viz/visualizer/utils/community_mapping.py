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


"""
These functions perform the multiple k means clusterings (using different k vals and initializations) and subsequent analysis including:
- reducing clusters into compositionally distinct sets using their ks statistic
- combining clusters into communities using louvain community detection
- describing each community with its mean ks statistic
- mapping a community identity to each k cluster
- assigning a community to each neighborhood by voting (mode)
The generation of any information necessary to create figures will also be done here, but the figures themselves will be created in the dashboard folder

"""

from utils.clustering import multiple_kmeans
import networkx as nx
import os.path as op
import json
import itertools
import numpy as np
import pandas as pd
from scipy.spatial import distance
import community as community_louvain


def community_mapping(filepath, kvals, n_repeats):
    """
    This will work for both single k and multiple k values. but now only really works for one because of file naming
    Might want to move the initial if statements into the routes.py and just pass in kvals
    Do we need to take the graph outside of the k for loop and change the naming of json files to include k?
    """

    # quantile neighborhood similarity cutoff for graph - can let users set this if desired
    q = 15

    df = pd.read_csv(filepath)
    if 'outer_layer' in list(df.columns.values):
        df = df.loc[df['outer_layer']==False]
    ion_types = sorted([c[0:] for c in df.columns if c[0]=='p'])
    
    n=0
    G=nx.Graph()
    for k in kvals:
        df_temp = multiple_kmeans(filepath, k, n_repeats)
        
        # add nodes to graph based on cluster
        for seed in range(n_repeats):
            df[f"kmeans{k}_{seed}"] = df_temp[f"kmeans{k}_{seed}"]
            
            # load json for individual clustering
            jsonfile = op.splitext(filepath)[0] + f'_{k}k_{seed}seed.json'
            with open(jsonfile) as f:
                data=json.load(f)

            # set embedding in order of ions
            for cluster in data['ks_statistics']:
                print(data['ions'])
                cluster_ks=dict(zip(data['ions'], data['ks_statistics'][cluster]))
                embedding_ks=[]
                for ion in [a[1:] for a in ion_types]:
                    if ion in cluster_ks.keys():
                        embedding_ks.append(cluster_ks[ion])
                    else:
                        embedding_ks.append(0)

                # add node for each cluster to graph
                # attrs: ks, seed, cluster, radius, overlap, count
                G.add_node(n, ks=embedding_ks, seed=seed, cluster=int(cluster), k_means=k,
                        N_mil=len(df.loc[df[f'kmeans{k}_{seed}']==int(cluster)])/1000000,
                        )
                for i,ion in enumerate([a[1:] for a in ion_types]):
                    nx.set_node_attributes(G, {n:round(embedding_ks[i],2)}, ion)
                n+=1


    # add edges based on euclidean distance between ks stats
    ks_embeddings=nx.get_node_attributes(G, 'ks')
    G.add_weighted_edges_from([(e[0][0],e[0][1],e[1]) for e in zip(itertools.combinations(range(len(G)), 2),
                                                                distance.pdist(list(ks_embeddings.values()), 'cosine'))])
    weights=np.array(list(nx.get_edge_attributes(G,'weight').values()))
    
    # I removed the G from the list of arguments to make it happy. I hope this is ok.
    def filter_edge(n1,n2):
        return G[n1][n2].get("filter",True)

    # remove weights above q-th percentile
    cutoff=np.percentile(weights, q=q)

    edge_filter = {(n1,n2):(False if attrs['weight']>cutoff else True) for n1,n2,attrs in G.edges(data=True)}

    nx.set_edge_attributes(G, edge_filter, 'filter')

    view = nx.subgraph_view(G, filter_edge=filter_edge)
    pos = nx.nx_agraph.graphviz_layout(view, prog='neato')

    G.remove_nodes_from([n for n,d in list(view.degree()) if d==0])
    view = nx.subgraph_view(G, filter_edge=filter_edge)

    ### find communities
    partition = community_louvain.best_partition(view, weight='weight', randomize=False, resolution=1, random_state=42)
    n_partitions=len(set(partition.values()))

    # get mean ks-statistics for each identified community
    embs=list(ks_embeddings.values())
    mean_ks = []
    for p in range(n_partitions):
        idxs=np.argwhere(np.array(list(partition.values()))==p).flatten()
        mean_ks.append(np.array([x for i,x in enumerate(embs) if i in idxs]).mean(axis=0))


    # get the mode of the partitions to identify the community of the neighborhood   
    def mode(lst):
        ## NB: if two partitions are equally represented as the mode 
        ##     the smaller value will be given
        return max(set(lst), key=lst.count)

    # partition dataframe used for mapping to neighborhood dataframe
    pf = pd.DataFrame({'node':partition.keys(),
                    'partition':partition.values()})
    pf['seed']=pf['node'].apply(lambda x: G.nodes[x]['seed'])
    pf['cluster']=pf['node'].apply(lambda x: G.nodes[x]['cluster'])
    pf['k_means']=pf['node'].apply(lambda x: G.nodes[x]['k_means'])


    # add mapping for removed clusters
    x=pf['seed'].value_counts()
    seeds_missing_cluster=list(x[x<k].index)

    missing_seeds=[]
    missing_clusters=[]
    for seed in seeds_missing_cluster:
        x=list(set(range(k))-set(pf.loc[pf['seed']==seed]['cluster'].tolist()))
        missing_seeds+=[seed]*len(x)
        missing_clusters+=x

    tmp = pd.DataFrame({'seed':missing_seeds, 'cluster':missing_clusters})
    tmp['node']=-1
    tmp['partition']=-1
    tmp['k_means']=k

    pf = pd.concat([pf,tmp],ignore_index=True)

    kmax = int(max(kvals))

    for s in range(n_repeats):
        partmap={row['cluster']:row['partition'] for i,row in pf.loc[(pf.seed==s)&(pf.k_means==kmax)].iterrows()}
        for k in range(kmax):
            if not k in partmap.keys():
                partmap[k]=-1
        df[f'partition{kmax}_{s}']=df[f'kmeans{kmax}_{s}'].apply(lambda x: int(partmap[x]))
    
    df['community']=df.apply(lambda row: mode([row[f"partition{kmax}_{s}"] for s in range(n_repeats)]), axis=1)
    # get the mode of the partitions to identify the community of the neighborhood   
    cols = [col for col in df.columns if f'partition{kmax}' in col]

    # percent agreement among clusterings
    df['agreement']=df.apply(lambda row: len(np.argwhere(row[cols].to_numpy()==row['community']))/len(cols), axis=1)



    # save clustering results
    csv_to_save = op.splitext(filepath)[0] + f'_clustered.csv'
    df.to_csv(csv_to_save, index=False)


    ebunch=[k for k, v in edge_filter.items() if v==False]
    G.remove_edges_from(ebunch)
    output_dict = {
        'G': nx.to_dict_of_dicts(G),
        "pos": pos,
        "partition": partition,
        "mean_ks": mean_ks,
        "n_partitions": n_partitions,
        "ion_types": ion_types,
        "clustered_filepath": csv_to_save
    }
    return output_dict
    
