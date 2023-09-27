import os
import h5py

from torch.utils.data import Dataset

dataset_path = os.path.dirname(__file__)

class ModelNet40(Dataset):
    """
    Dataset class for ModelNet40 from Wu et al. "3D ShapeNets: A Deep 
    Representation for Volumetric Shapes", CVPR, 2015
    
    Args:
        ...
    """
    def __init__(self):
        metadata = getMetadata()
        self.point_cloud_data = getPointCloud()
    
    def loadDataset(self):
        if os.path.exists(os.path.join(dataset_path, 'modelnet40_ply_hdf5_2048')):
            print('Dataset already exists!')
            return
        
        dataset_link = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile_name = os.path.basename(dataset_link)
        
        print('Downloading dataset ...\n')
        os.system(f'wget {dataset_link} --no-check-certificate')
        os.system(f'unzip {zipfile_name} -d {dataset_path}')
        os.system('rm %s' % (zipfile_name))
        print(f'Download complete: {dataset_path + zipfile_name[:-4]}')
    
    def __len__(self):
        ...
    
    def __getitem__(self, index):
        ...

def getMetadata():
    dataset_path = os.path.dirname(__file__)
    # check whether the metadata file exists
    metadata_path = os.path.join(dataset_path, "metadata_modelnet40.csv")
    if os.path.isfile(metadata_path):
        print("File found")
        return 
    # if so, return the metadata file
    # else, create metadata file from data

def getPointCloud():
    ...

if __name__ == "__main__":
    dataset = ModelNet40()
    dataset.loadDataset()
