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
from sklearn import cluster


def create_dataframe(filepath):
    """
    Simply reads a csv and returns it as a dataframe
    """
    df = pd.read_csv(filepath)
    #count_col = 'N_atoms'
    #df.fillna(0, inplace=True)
    return df


def cluster_this(df):
    """
    Clusters a dataframe using kmeans and adds the cluster results to the dataframe as an additional column labeled 'kmeans'.
    This can be expanded to accomodate different types of clustering methods later.
    TODO: The clustering occurs in the routes.py, using functions from clustering.py. This is not currently used.
    """
    atom_types = [c[0:] for c in df.columns if c[0] == 'p']
    data = df[atom_types].to_numpy()  # .astype('float32')
    n_clusters = 5  # choice: n_clusters: number of clusters
    tag = 'kmeans'
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(data)
    df[tag] = kmeans.labels_
    df[tag] = df[tag].astype("category")
    #df.to_csv('data/clustered_data.csv')
    return df


def create_marks(df):
    """
    Creates an entry in a dictionary for each unique z value with an empty string as its value.
    """
    marks_dict = {}
    for z_value in df['midpoint_z'].unique():
        marks_dict[z_value] = ''
    return marks_dict


def collect_z_slice(z, df):
    """
    Filter a dataframe by z value and return the resulting slice of the data.
    """
    z_df = df.loc[df['midpoint_z'] == z]
    return z_df
