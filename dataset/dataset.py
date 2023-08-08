import pandas as pd
# import dask.dataframe as dd
import numpy as np
import torch

from torch.utils.data import Dataset

class MN40(Dataset):
    '''
    ModelNet40 Dataset
    Args:
        split: [None|train|test]
        num_sample: [1024|int] Number of sampled point clouds from mesh surfaces
    '''    
    def __init__(self, split=None, num_sample=1024):
        self.meta = pd.read_csv('dataset/metadata_modelnet40.csv')
        self.num_sample = num_sample
        
        if split in ('train', 'test'):
            self.meta = self.meta[self.meta['split']==split]
        elif split != None:
            raise Exception('ArgsError: Unexpected argument.')
        
    def __len__(self):
        return len(self.meta)
        
    def __getitem__(self, idx):
        cls = self.meta.iloc[idx, 1]
        off_name = self.meta.iloc[idx, 3]
        
        # Get pointcloud
        verts, faces = read_off('dataset/ModelNet40/' + off_name)
        cloud = sample_off(self.num_sample, verts, faces)
        
        # Normalize
        cloud = norm_cloud(cloud)
        
        return dict(cls=cls, cloud=cloud)
        
def read_off(filepath):
    '''
        Read .off file and return vertices and faces
    '''
    with open(filepath) as off:
        if 'OFF' not in off.readline():
            raise Exception('FileError: Invalid .off file.')
        
        num_vert, num_face, _ = map(int, off.readline().split())
        verts = [np.array(off.readline().split(), dtype='float32') for i in range(num_vert)]
        faces = [np.array(off.readline().split(), dtype='int') for i in range(num_face)]
        
        return verts, faces
    
def sample_off(num_sample, verts, faces):
    '''
        Input: Vertices, faces of a mesh
        Output: Sampled point cloud data
        
        1. sample meshes uniformly
        2. take a single point from the mesh surface
    '''
    def pick_point(face, verts):
        num_points = face[0]
        mesh = [verts[i] for i in face[1:]]
        point = mesh[0]
        for i in range(num_points - 1):
            t = torch.rand(1)
            point = t * point + (1 - t) * mesh[i + 1]
        return point
        
    sample_list = np.random.choice(len(faces), num_sample)
    cloud = torch.stack([pick_point(faces[i], verts) for i in sample_list])
    
    return cloud

def norm_cloud(cloud):
    '''
        Normalize input pointcloud into a unit sphere
    '''
    # Move to center
    mean = torch.mean(cloud, dim=0)
    cloud = cloud - mean.repeat(cloud.shape[0], 1)
    
    # Get max norm
    max_norm = max(np.linalg.norm(cloud, axis=1))
    cloud = cloud / max_norm
    
    return cloud
        
if __name__ == '__main__':
    # dataset test
    dataset_train = MN40(split='train')
    dataset_test = MN40(split='test')
    
    print(f'Train dataset size: {len(dataset_train)}')
    print(f'Test dataset size: {len(dataset_test)}')
    
    item = dataset_test[2400]
    print(item['cls'])
    print(len(item['cloud']))
    print(item['cloud'][:10])