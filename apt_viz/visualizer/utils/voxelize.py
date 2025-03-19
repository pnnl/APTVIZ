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
from scipy.spatial import cKDTree
from multiprocessing import Pool
from utils import unpack


class VoxelAPT:
    def __init__(self, posfile, rrngfile):
        # extract APT data from pos files
        self.coords, self.xyzids, self.atom_types = unpack.extract_APT_data(posfile, rrngfile, simplify=False)
    
    def load_voxels(self, savefile):
        self.df=pd.read_csv(savefile)
    
    def voxelize(self, box_length, N=12):
        """Voxelizes APT point cloud data
        Args:
            box_length (int): Length of voxel edge
            N (int): Number of CPU processors to use for multiprocessing
            
        Returns:
            df (pd.DataFrame): Dataframe of information for each voxel (column descriptions as follows) --
                               voxel (voxel index), voxel_length (voxel edge length), midpoint_x (central x coordiante), 
                               midpoint_y (central y coordinate), midpoint_z (central z coordinate), N_atoms (number of atoms in voxel),
                               ...percentage of each element in voxel...
        """
        self.box_length = box_length
        self._get_voxel_bounds()
        
        with Pool(N) as p:
            rows = p.map(self._voxel_dataframe, list(range(len(self.voxel_bounds))))

        df=pd.concat(rows, ignore_index=True)

        for atom in self.atom_types:
            df[atom]=df[atom].fillna(0)

        self.df=df[['voxel', 'voxel_length', 'midpoint_x', 'midpoint_y', 'midpoint_z', 'N_atoms']+self.atom_types]
        print(f'{len(self.df)} non-empty voxels')
        
    def _get_voxel_bounds(self):
        """Obtains bounds for each voxel        
        Returns:
            voxel_bounds (list): Bounding coordinates for each voxel
        """
        coords_min = np.around(self.coords.min(axis=0), decimals=-1)
        coords_max = np.around(self.coords.max(axis=0), decimals=-1)

        # get groups of bounds for each coordinate
        bounds=[]
        for i in range(len(coords_min)):
            #clist=list(range(int(coords_min[i]), int(coords_max[i])+1, self.box_length))
            clist=np.arange(int(coords_min[i]), int(coords_max[i]), self.box_length)
            bounds.append([(clist[r],clist[r+1]) for r in range(len(clist)-1)])

        # pair up different voxel bounds
        self.voxel_bounds=list(itertools.product(bounds[0],bounds[1],bounds[2]))
        
    def _voxel_dataframe(self, vox):
        """Extract information on ions for each voxel
        Args:
            vox (int): Index of voxel to analyze
        
        Returns:
            row (pd.DataFrame): Row in dataframe for each voxel
        """
        vox_ids_x=np.where((self.coords.T[0]>=self.voxel_bounds[vox][0][0])&(self.coords.T[0]<=self.voxel_bounds[vox][0][1]))[0]
        vox_ids_y=np.where((self.coords.T[1]>=self.voxel_bounds[vox][1][0])&(self.coords.T[1]<=self.voxel_bounds[vox][1][1]))[0]
        vox_ids_z=np.where((self.coords.T[2]>=self.voxel_bounds[vox][2][0])&(self.coords.T[2]<=self.voxel_bounds[vox][2][1]))[0]

        vox_idx=list(set(vox_ids_x).intersection(vox_ids_y).intersection(vox_ids_z))

        if len(vox_idx) == 0:
            return 
        
        vox_ids = self.xyzids[vox_idx]
        #vox_coords = self.coords[vox_idx]
        #write_xyz(vox_coords, vox_ids, savefile=f'voxel{vox}.xyz')

        row=pd.Series(vox_ids).value_counts()
        row=row.to_frame()
        #turn into percent
        row=100*row/len(vox_ids)
        row=row.T
        row['voxel']=vox
        row['voxel_length']=self.box_length
        row['N_atoms']=len(vox_ids)
        row['midpoint_x']=np.mean(self.voxel_bounds[vox][0])
        row['midpoint_y']=np.mean(self.voxel_bounds[vox][1])
        row['midpoint_z']=np.mean(self.voxel_bounds[vox][2])

        return row



def create_mesh(start, stop, num=10):
    # create meshgrid
    mesh, step = np.linspace(start, stop, num, retstep=True)
    xx, yy, zz = np.meshgrid(mesh, mesh, mesh, sparse=False)

    # reshape and flatten mesh coords
    mesh = np.array(list(zip(xx.flatten(), yy.flatten(), zz.flatten())))
    mesh_tree = cKDTree(mesh)

    # create a dataframe for the mesh
    mesh_df = pd.DataFrame({'mesh':range(len(xx.flatten())), 'x':xx.flatten(), 'y':yy.flatten(), 'z':zz.flatten()})
    
    return mesh_tree, mesh_df, step

def attach_ion_to_mesh(ion_df, mesh_tree, ion, step):
   
    
    # set up single ion KDTree
    tree = cKDTree(list(zip(ion_df['x_norm'],ion_df['y_norm'],ion_df['z_norm'])))
    
    # pair ions with mesh points
    indexes = tree.query_ball_tree(mesh_tree, r=step/2)
    
    # get counts from pair list
    pair_list = [[(i,j) for j in indexes[i]] for i in range(len(indexes))]
    pair_list = [x for x in pair_list if x!=[]]
    
    # put list into dataframe
    pf = pd.DataFrame({'mesh':np.array(pair_list).squeeze().T[1], 'ion': np.array(pair_list).squeeze().T[0]})
    
    # get counts of ions per mesh spacing
    mf = pd.DataFrame({'mesh':pf.mesh.value_counts().index, f'{ion}':pf.mesh.value_counts().values})

    return mf





