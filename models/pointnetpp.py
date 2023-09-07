import random

import torch
import torch.nn as nn

class PointNetpp(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.set_abstraction_layers = nn.ModuleList([])

    def forward(self, x):
        return x

class SetAbstractionLayer(nn.Module):
    """
    Single set abstraction layer of PointNet++ architecture.
        
    Args:
        ...
    """
    def __init__(
        self
    ):
        super().__init__()
        
        ...

    def forward(
        self,
        point_coord: torch.Tensor,
        features: torch.Tensor
    ):
        """
        Args:
            point_coord: (B, N, 3) tensor
            features: (B, N, C) tensor
        
        Returns:
            point_coord: (B, N, 3) tensor containing point clouds' xyz coordinates.
            features: (B, N, C) tensor
            
        Todo:
            Sampling layer: use farthest point sampling(FPS) to get subset of points.
            Grouping layer: use ball query algorithm to get N' clusters of points.
                Each clusters have different number of points upper-limited to K.
            PointNet layer: each clusters are processed within PointNet-like module
                and the output is abstracted by its centroid and local feature.
        """
        
        
        return ...
    
@torch.no_grad()
def FPS(
    points: torch.Tensor,
    num_sample: int
) -> torch.Tensor:
    '''
        Farthest Point Sampling(FPS)
        Finds indices of a subset of points that are the farthest away from each other.
        [B, N, 3] -> [B, M]
        Source code courtesy: pytorch3d docs
    '''
    num_batch, num_points, _ = points.size()
    device = points.device
    
    sampled_indices = []    
    for batch in range(num_batch):
        sample_idx = torch.full((num_sample, ), -1, dtype=torch.int64, device=device)
        closest_distances = points.new_full((num_points, ), float("inf"), dtype=torch.float32)
        selected_idx = random.randint(0, num_points)
        sample_idx[0] = selected_idx
        
        for i in range(num_sample):
            distance = points[batch, selected_idx, :] - points[batch, :num_points, :]
            distance = (distance**2).sum(-1)
            
            closest_distances = torch.min(distance, closest_distances)
            
            selected_idx = torch.argmax(closest_distances)
            sample_idx[i] = selected_idx
        
        sampled_indices.append(sample_idx)
    sampled_indices = torch.stack(sampled_indices)
    
    return sampled_indices

@torch.no_grad()
def ball_query(
    points: torch.Tensor,
    centroid_indices: torch.Tensor,
    radius: float
) -> list:
    '''
        Ball query clustering.
        Returns list of indices tensor grouped by each clusters
        [N, 3], [M] -> [M, [K]]
    '''
    cluster_list = []
    
    for centroid_index in centroid_indices:
        centroid = points[centroid_index, :]
        distance = centroid - points
        distance = torch.abs((distance**2).sum(-1))
        
        cluster_indices = [i for i in range(len(distance)) if distance[i] <= radius]
        cluster_indices = torch.Tensor(cluster_indices).type(torch.int64)
        cluster_list.append(cluster_indices)
        
    return cluster_list

if __name__ == "__main__":
    # points = torch.rand([10, 1024, 3])
    # sampled_indices = FPS(points, 10)
    # print(points[sampled_indices].shape)
    
    # for indices in sampled_indices:
    #     print(indices)
    
    points = torch.rand((10, 3))
    centroid_indices = torch.randint(0, 10, (5, ))
    radius = 0.3
    
    print(ball_query(points, centroid_indices, radius))
    