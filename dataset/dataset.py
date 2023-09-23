import os

from torch.utils.data import Dataset

class ModelNet40(Dataset):
    """
    Dataset class for ModelNet40 from Wu et al. "3D ShapeNets: A Deep 
    Representation for Volumetric Shapes", CVPR, 2015
    
    Args:
        ...
    """
    def __init__(self):
        metadata = getMetadata()
        self.point_cloud_data = getPointCloud(metadata)
    
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
    getMetadata()
