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
import numpy as np
import seaborn as sns
import community as community_louvain
from utils import graph
import matplotlib
import time

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--datadir', required=True, type=str, help='Location of APT neighborhood files.')
parser.add_argument('-r','--radius', required=True, type=float, help='Radius of neighborhood.')
parser.add_argument('-o','--overlap', required=True, type=float, help='Neighborhood overlap.')
parser.add_argument('-k','--k',  required=True, nargs='+', help='List of k-means clusters.')
parser.add_argument('-s','--seeds', required=True, nargs='+', help='List of seeds for k-means clustering.')
parser.add_argument('-q','--q', default=15, type=float, help='Remove graph edges with weights above the qth percentile.')
args = parser.parse_args()

start = time.time()

savedir=op.join(args.datadir, 'processed')
sim_scores_file = op.join(savedir,f'kmeans_cluster_similarity_{args.radius}nm-radius_{args.overlap}-overlap.csv')
sim_graph_json = op.join(savedir,f'kmeans_cluster_graph_{args.radius}nm-radius_{args.overlap}-overlap.json')


## run community clustering
G, ions=graph.read_json_file(sim_graph_json)
weights=np.array(list(nx.get_edge_attributes(G,'weight').values()))
ks_embeddings=nx.get_node_attributes(G, 'ks')

def filter_edge(n1,n2, G=G):
    return G[n1][n2].get("filter",True)

# remove weights above q-th percentile
cutoff=np.percentile(weights, q=args.q)
print(f'Edges with similarity <{cutoff} removed.')

edge_filter = {(n1,n2):(False if attrs['weight']>cutoff else True) for n1,n2,attrs in G.edges(data=True)}
nx.set_edge_attributes(G, edge_filter, 'filter')
view = nx.subgraph_view(G, filter_edge=filter_edge)
pos = nx.nx_agraph.graphviz_layout(view, prog='neato')

print(f'{len([n for n,d in list(view.degree()) if d==0])} isolated nodes removed')
G.remove_nodes_from([n for n,d in list(view.degree()) if d==0])
view = nx.subgraph_view(G, filter_edge=filter_edge)

### find communities
partition = community_louvain.best_partition(view, weight='weight', randomize=False, resolution=1, random_state=42)
n_partitions=len(set(partition.values()))
print(f'{n_partitions} partitions found')

graph.plot_partitions(view, pos, partition, weights, saveas=op.join(savedir, f'kmeans_cluster_similarity_{args.radius}nm-radius_{args.overlap}-communities-{args.q}percentile.png'))


### color kmeans clusters belonging to community
def plot_graph(G, pos, node_colors, edge_weights, saveas=''):

    # color the nodes according to their partition
    node_colors = [int(k) for k in node_colors]
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
    cbar.ax.set_yticks([x+0.5 for x in range(n_colors)],range(4,4+n_colors))
    cbar.set_label(label='k-means', size=12)
    ax[0].axis('off')
    ax[1].axis('off')
    plt.tight_layout()
    if len(saveas)>0:
        plt.savefig(saveas, dpi=300)
    plt.close()

df = pd.DataFrame({'node':partition.keys(), 'partition':partition.values()})
df['k'] = df['node'].apply(lambda n: G.nodes[n]['k_means'])
df['seed'] = df['node'].apply(lambda n: G.nodes[n]['seed'])
df['cluster'] = df['node'].apply(lambda n: G.nodes[n]['cluster'])
plot_graph(view, pos, df.k.tolist(), weights, saveas=op.join(savedir, f'kmeans_cluster_similarity_{args.radius}nm-radius_{args.overlap}-communities-by-kmeans-{args.q}percentile.png'))


# get mean ks-statistics for each identified community
embs=list(ks_embeddings.values())
mean_ks = []
for p in range(n_partitions):
    idxs=np.argwhere(np.array(list(partition.values()))==p).flatten()
    mean_ks.append(np.array([x for i,x in enumerate(embs) if i in idxs]).mean(axis=0))
    
plt.figure(figsize=(8,4))
plt.title('Mean KS Statistic')
sns.heatmap(mean_ks, square=False, yticklabels=range(1,1+n_partitions), 
            xticklabels=[graph.format_text(ion) for ion in ions], cmap='bwr', vmin=-1, vmax=1,)
plt.ylabel('Community', fontsize=12)
plt.tight_layout()
plt.savefig(op.join(savedir, f'kmeans_cluster_similarity_{args.radius}nm-radius_{args.overlap}-communities-meanksstats-{args.q}percentile.png'), dpi=300)
plt.close()



## map clusters to partitions
if not op.isdir(op.join(args.datadir,'partition_map')):
    os.mkdir(op.join(args.datadir,'partition_map'))

# get the mode of the partitions to identify the community of the neighborhood   
def mode(lst):
    ## NB: if two partitions are equally represented as the mode 
    ##     the smaller value will be given
    return max(set(lst), key=lst.count)

for n_clusters in args.k:
    file = op.join(args.datadir, f'kmeans_over_all_{args.radius}nm-radius_{args.overlap}-overlap_{n_clusters}clusters.csv')

    if op.isfile(file):
        print(f'Mapping from clusters from {file}')
        allcf = pd.read_csv(file, usecols=['sample','midpoint_x','midpoint_y','midpoint_z']+[f'kmeans{s}' for s in args.seeds]) 
    else:
        print(f'Missing cluster file -- skipping {file}')

    samples = list(allcf['sample'].value_counts().index)

    for samp in samples:
        part = op.join(args.datadir,'partition_map', f"{samp}_partitions_{n_clusters}-kmeans_{args.radius}nm-radius_{args.overlap}-overlap-{args.q}percentile.csv")

        if not op.isfile(part):
            print(f'...{samp}')
            # if community merge file isn't written, write it

            cf = allcf.loc[allcf['sample']==samp].copy()

            for s in args.seeds:
                partmap={row['cluster']:row['partition'] for i,row in df.loc[(df.seed==s)&(df.k==n_clusters)].iterrows()}
                for k in range(int(n_clusters)):
                    if not k in partmap.keys():
                        partmap[k]=-1
                cf[f'part{s}']=cf[f'kmeans{s}'].apply(lambda x: partmap[x])

            cf['community']=cf.apply(lambda row: mode([row[f"part{s}"] for s in args.seeds]), axis=1)
            cf.to_csv(part, index=False)


for k in args.k:
    allcf=pd.DataFrame()
    for samp in samples:

        mapped_file = op.join(args.datadir,'partition_map', 
                              f"{samp}_partitions_{k}-kmeans_{args.radius}nm-radius_{args.overlap}-overlap-{args.q}percentile.csv")
        cf = pd.read_csv(mapped_file,  usecols=['sample','midpoint_x','midpoint_y','midpoint_z','community'])
        cf['community']+=1
        allcf = pd.concat([allcf, cf], ignore_index=True, sort=False)

        # write to XYZ files for plotting with external programs
        graph.write_xyz(cf, mapped_file.replace('.csv','.xyz'), tag='community')

    allcf.to_csv(op.join(args.datadir,'partition_map',
                        f"all_partitions_{k}-kmeans_{args.radius}nm-radius_{args.overlap}-overlap-{args.q}percentile.csv"), index=False)


print(f"Community mapping finished in {(time.time()-start)/60:0.2f} min")
