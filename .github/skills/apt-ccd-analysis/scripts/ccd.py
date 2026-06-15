from scipy import spatial
from pathlib import Path
import numpy as np
import pandas as pd
import os
import os.path as op
import matplotlib.pyplot as plt
from scipy import spatial
from . import unpack
from .OPTICSAPT.APTPosData import APTPosData
from collections import Counter
import csv 
import json
from sklearn.cluster import MiniBatchKMeans
import networkx as nx
import itertools
import community as community_louvain
import seaborn as sns
import scipy.stats as st  # requires scipy >= 1.9 (statistic_sign)
from scipy.spatial import distance
import logging


# --- Tool Function Factory ---

def create_tool_functions(file):
    return {"generate_neighborhoods": lambda rrng=None, radius=1, overlap=0.5: generate_neighborhoods(file, rrng, radius, overlap),
            "detect_compositional_communities": lambda neighborhood_data, savedir='', k_values=[4,5,6], ignore_ions=['O1', 'O1H1', 'O2'], n_repeats=2, q=25: detect_compositional_communities(neighborhood_data, savedir, k_values, ignore_ions, n_repeats, q)}

# --- APT Tool Definitions (OpenAI format) ---

tool_schema = [{
        "type": "function",
        "function": {
            "name": "generate_neighborhoods",
            "description": "Generate overlapping spherical neighborhoods of ions from reconstructed APT data." \
                           "Returns neighborhood data and writes it to a .csv." \
                           "Works for APT data. Requires both .pos file and .rrng file for data extraction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "radius": {
                        "type": "number",
                        "description": "Radius of the neighborhood in microns (default: 1.0)"
                    },
                    "overlap": {
                        "type": "number",
                        "description": "Overlap fraction between neighborhoods (default: 0.5)"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "detect_compositional_communities",
            "description": "Detect compositional communities in neighborhood data from reconstructed APT data." \
                           "Returns community data and writes it to a .csv." \
                           "Works for APT data. Requires neighborhood data generated from generate_neighborhoods function.",
            "parameters": {
                "type": "object",
                "properties": {
                    "k_values": {
                        "type": "array",
                        "items": {
                            "type": "number"
                        },
                        "description": "List of k values for community detection (default: [4,5,6])"
                    },
                    "n_repeats": {
                        "type": "number",
                        "description": "Number of repeats for each k value (default: 2)"
                    },
                    "ignore_ions": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "List of ions to ignore during community detection (default: ['O1', 'O1H1', 'O2'])"
                    },
                    "q": {
                        "type": "number",
                        "description": "Resolution parameter for community detection (default: 25)"
                    }
                },
                "required": []
            }
        }
    }]


def generate_neighborhoods(file, rrng, savedir='ccd_analysis', 
                           csv_column_labels=['x','y','z','Da'],
                           radius=1, overlap=0.5):
    # Extract sample name
    sample = Path(file).stem

    # Set up directory and file naming convention
    os.makedirs(savedir, exist_ok=True)
    save_file=op.join(savedir, f"{sample}_{radius}nm-radius_{overlap}-overlap.csv")

    # Extract APT data
    if Path(file).suffix.lower() not in ['.pos', '.apt', '.csv']:
        raise ValueError("Unsupported file type. Please provide a .pos, .apt, or .csv file.")
    elif Path(file).suffix.lower() == '.csv':
        df = pd.read_csv(file, names=csv_column_labels)
        coords = df[['x','y','z']].to_numpy()
        das = df['Da'].tolist()
        ## Infer ion types from Da values using rrng file
        _pos_data = APTPosData()
        _pos_data.load_range_file(rrng)
        _fake_pos = np.column_stack([coords, np.array(das)])
        xyzids = _pos_data.range_pos(_fake_pos)

        # remove noise
        noise_idx = np.where(np.array(xyzids)=='Noise')
        coords = np.delete(coords, noise_idx, 0)
        xyzids = np.delete(xyzids, noise_idx, 0)
        atom_types = sorted(list(set(xyzids)))
    else:
        coords, xyzids, atom_types = unpack.extract_APT_data(file, rrng, simplify=False)

    #print(f"{len(coords)} ions")

    # KDTree for neighborhood collection
    tree = spatial.KDTree(coords)

    # extract coordinates
    X,Y,Z = coords.T

    # calculate grid spacing
    step = (2*radius)*(1-overlap)

    # create meshgrid
    xx, yy, zz = np.meshgrid(np.arange(X.min()+radius, X.max()-radius+(step/2), step), 
                            np.arange(Y.min()+radius, Y.max()-radius+(step/2), step), 
                            np.arange(Z.min()+radius, Z.max()-radius+(step/2), step), sparse=False)

    mesh = np.array(list(zip(xx.flatten(), yy.flatten(), zz.flatten())))
    mesh_tree = spatial.KDTree(mesh)

    # match mesh points to ion tree
    matches=mesh_tree.query_ball_tree(tree, r=radius)

    # collect only non-empty voxels
    nonempty_matches=[(idx, mtch) for idx, mtch in enumerate(matches) if len(mtch)>0]

    field_names=atom_types+['p'+a for a in atom_types]+['d'+a for a in atom_types]+['N_atoms','density','midpoint_x','midpoint_y','midpoint_z','radius','id']

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

    # write header
    with open(save_file, 'w') as csv_file:
        dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
        dict_object.writerow({i:i for i in field_names})

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

    return {
            "sample": sample,
            "range_file": rrng,
            "neighborhood_radius": radius,
            "neighborhood_overlap": overlap,
            "neighborhood_count": len(df),
            "ion_count": len(atom_types),
            "mean_neighborhood_density": df.loc[~df['outer_layer']]['density'].mean(),
            "std_neighborhood_density": df.loc[~df['outer_layer']]['density'].std(),
            "ion_type_counts": Counter(xyzids),
            #"neighborhoods": df.to_json(orient="index")
        }

def write_xyz(df, savefile, community='community'):
    coords = ['   '.join([str(int(c[0]))] + [str(v) for v in c[1:]] + ['\n'])
              for c in np.array(df[[community,'midpoint_x','midpoint_y','midpoint_z']])]
    with open(savefile, "w") as f:
        f.writelines([str(len(coords))+'\n', " \n"])
        f.writelines(coords)

def format_text(text):
    if text[-1]=='1':
        text = text[:-1]+'^{{+}}'
    for i in range(2,5):
        i=str(i)
        if text[-1]==i:
            text = text[:-1]+'^{'+i+'{+}}'
    text=text.replace('1','').replace('2','_{2}').replace('3','^{3}').replace('4','^{4}')
    return r"$\mathrm{"+text+r"}$"

def detect_compositional_communities(neighborhood_data, 
                                     savedir='ccd_analysis', 
                                     k_values=[4,5,6],
                                     ignore_ions=[],
                                     n_repeats=2, 
                                     q=25):
    # save info
    sample_name = Path(neighborhood_data).stem
    savefile = f'{sample_name}_community_clustering'

    df = pd.read_csv(neighborhood_data)

    # remove neighborhoods in the outer layer
    df = df.loc[df['outer_layer']==False].copy()

    # infer ion types
    ion_types = sorted([c[0:] for c in df.columns if c[0]=='p'])

    # Remove specified ions from analysis
    for i in ignore_ions:
        ion_types.remove('p'+i)

    # recalculate percentage (pX) using only the specified ions

    # get number of atoms in neighbood
    df['N_atoms_subset']=df[[ion[1:] for ion in ion_types]].sum(axis=1)

    # drop neighborhoods with no ions left
    df = df.loc[df['N_atoms_subset']>0].copy()

    #for ion in ion_types:
    #    df[ion] = df.apply(lambda row: row[ion[1:]]/row['N_atoms_subset'] , axis=1)

    # extract data to cluster
    data = df[ion_types].to_numpy()

    # Run KMeans clustering for each k and collect KS statistics for each cluster
    for k in k_values:
        for seed in range(n_repeats):
            # run kmeans clustering
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=10).fit(data)
            df[f"{k}kmeans{seed}"]=kmeans.labels_
            df[f"{k}kmeans{seed}"]=df[f"{k}kmeans{seed}"].astype("category")

            # calculate ks statistics for each cluster
            ks_stats=[]
            for name, group in df.groupby(f"{k}kmeans{seed}", observed=True)[ion_types]:
                
                def get_stat(ion, bulk=df, group=group):
                    stats=st.ks_2samp(bulk[ion], group[ion])
                    return stats.statistic_sign*stats.statistic
                
                stats=[get_stat(ion, df, group) for ion in ion_types]
                    
                ks_stats.append(stats)

            # save metadata as .json
            jsonsavefile = op.join(savedir, savefile+f'_{k}_{seed}seed')
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

            with open(jsonsavefile+".json", "w") as outfile:
                json.dump(metadata, outfile)


    # add nodes to graph based on cluster
    n=0
    G=nx.Graph()
    for k in k_values:
        for seed in range(n_repeats):
            
            # load json for individual clustering
            with open(os.path.join(savedir,savefile+f'_{k}_{seed}seed.json')) as f:
                data=json.load(f)

            # set embedding in order of ions
            for cluster in data['ks_statistics']:
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
                        N_mil=len(df.loc[df[f'{k}kmeans{seed}']==int(cluster)])/1000000,
                        )
                for i,ion in enumerate([a[1:] for a in ion_types]):
                    nx.set_node_attributes(G, {n:round(embedding_ks[i],2)}, ion)
                n+=1


    # add edges based on euclidean distance between ks stats
    ks_embeddings=nx.get_node_attributes(G, 'ks')
    G.add_weighted_edges_from([(e[0][0],e[0][1],e[1]) for e in zip(itertools.combinations(range(len(G)), 2),
                                                                distance.pdist(list(ks_embeddings.values()), 'cosine'))])
    weights=np.array(list(nx.get_edge_attributes(G,'weight').values()))

    def filter_edge(n1,n2, G=G):
        return G[n1][n2].get("filter",True)

    # remove weights above q-th percentile
    cutoff=np.percentile(weights, q=q)

    edge_filter = {(n1,n2):(False if attrs['weight']>cutoff else True) for n1,n2,attrs in G.edges(data=True)}
    nx.set_edge_attributes(G, edge_filter, 'filter')

    view = nx.subgraph_view(G, filter_edge=filter_edge)

    #print(len([n for n,d in list(view.degree()) if d==0]), ' isolated nodes')
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


    ## TODO: somewhere after here is where divergence between the apt and scilink versions starts to occur.

    # partition dataframe used for mapping to neighborhood dataframe
    pf = pd.DataFrame({'node':partition.keys(),
                    'partition':partition.values()})
    pf['seed']=pf['node'].apply(lambda x: G.nodes[x]['seed'])
    pf['cluster']=pf['node'].apply(lambda x: G.nodes[x]['cluster'])
    pf['k_means']=pf['node'].apply(lambda x: G.nodes[x]['k_means'])

    # map cluster id to partition — rebuild lookup per k so cross-k rows don't pollute lookups
    for k in k_values:
        pf_k = pf.loc[pf['k_means']==k].copy()

        # pad any (seed, cluster) pairs absent from the graph (isolated-node removals)
        sv = pf_k['seed'].value_counts()
        missing_seeds, missing_clusters = [], []
        for seed in sv[sv < k].index:
            missing = list(set(range(k)) - set(pf_k.loc[pf_k['seed']==seed]['cluster'].tolist()))
            missing_seeds += [seed] * len(missing)
            missing_clusters += missing
        if missing_seeds:
            tmp_k = pd.DataFrame({'seed': missing_seeds, 'cluster': missing_clusters,
                                  'node': -1, 'partition': -1, 'k_means': k})
            pf_k = pd.concat([pf_k, tmp_k], ignore_index=True)

        for i in range(n_repeats):
            mapping = pf_k.loc[pf_k['seed']==i].set_index('cluster')['partition'].to_dict()
            df[f'partition{i}'] = df[f'{k}kmeans{i}'].astype(int).map(mapping)

    # get the mode of the partitions to identify the community of the neighborhood   
    cols = [col for col in df.columns if 'partition' in col]
    # convert to ints
    df[cols]=df[cols].astype(np.int16)

    df['community']=df[cols].mode(axis='columns')[0].astype(np.int16)

    write_xyz(df, op.join(savedir, savefile+".xyz"), community='community')

    # plot 
    plt.figure(figsize=(n_partitions, len(ion_types)/2.5))
    plt.title('Mean KS Statistic')
    ax=sns.heatmap(np.vstack(mean_ks).T, square=False, #xticklabels=range(1,1+n_partitions), 
                yticklabels=[format_text(ion[1:]) for ion in ion_types], cmap='bwr', vmin=-1, vmax=1,)
    plt.xlabel('Community', fontsize=12)
    ax.tick_params(axis='y', labelrotation=0)
    plt.tight_layout()
    plt.savefig(op.join(savedir, 'KS_stats.png'), dpi=300)
    plt.close()

    return {
            "community_count": n_partitions,
            "community_neighborhood_counts": Counter(df['community'].tolist()),
            "community_compositions": mean_ks,
           }




