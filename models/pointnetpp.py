import random

import torch
import torch.nn as nn
from torch.autograd import Function

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
        num_sample: the number of points to be sampled by sampler.
    """
    def __init__(
        self,
        num_sample: int
    ):
        super().__init__()
        
        self.num_sample = num_sample
        self.sampling = FarthestPointSampling.apply
        self.gather = GatherPoints.apply

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
        centroid_indices = self.sampling(point_coord, self.num_sample)
        centroid_coord = self.gather(point_coord, centroid_indices)
        
        return centroid_coord

class FarthestPointSampling(Function):
    """
    Custom function implementation of farthest point sampling(FPS).
    Wrapping farthestPointSampling function.
    
    Args:
        point_coord: (B, N, 3) tensor
        num_sample: number of sampled points
    Returns:
        sample_indices: (B, num_sample) tensor of indices
    """
    @staticmethod
    def forward(
        ctx,
        point_coord: torch.Tensor,
        num_sample: int
    ) -> torch.Tensor:
        sample_indices = farthestPointSampling(point_coord, num_sample)
        ctx.mark_non_differentiable(sample_indices)
        return sample_indices
    
    @staticmethod
    def backward(ctx, g=None):
        return None, None

class GatherPoints(Function):
    """
    Custom function for slicing point cloud tensors according to given indices.
    
    Args:
        points: (B, N, C) tensor
        indices: (B, N') tensor containing target indices.
    Returns:
        new_points: (B, N', C) tensor
    """
    @staticmethod
    def forward(
        ctx,
        points: torch.Tensor,
        indices: torch.Tensor
    ):
        new_points_list = []
        for b, indices_batch in enumerate(indices):
            new_points_list_batch = [points[b, index, :] for index in indices_batch]
            new_points_batch = torch.stack(new_points_list_batch)
            
            new_points_list.append(new_points_batch)
        
        B, N, C = points.shape
        ctx.for_backwards = (indices, B, N, C)
        
        new_points = torch.stack(new_points_list)
        return new_points
    
    @staticmethod
    def backward(ctx, grad_output):
        indices, B, N, C = ctx.for_backwards
        grad_features = torch.zeros((B, N, C))
        
        for b in range(B):
            for i, index in enumerate(indices[b]):
                grad_features[b, index] = grad_output[b, i]
        
        return grad_features, None

def farthestPointSampling(
    point_coord: torch.Tensor,
    num_sample: int
) -> torch.Tensor:
    """
    Finds indices of a subset of points that are the farthest away from each other.
    Source code courtesy: pytorch3d docs
    
    Args:
        point_coord: (B, N, 3) tensor
        num_sample: number of sampled points
    Returns:
        sample_indices: (B, num_sample) tensor of indices
    """
    num_batch, num_points, _ = point_coord.size()
    device = point_coord.device
    
    sample_indices = []    
    for batch in range(num_batch):
        sample_idx = torch.full((num_sample, ), -1, dtype=torch.int64, device=device)
        closest_distances = point_coord.new_full((num_points, ), float("inf"), dtype=torch.float32)
        selected_idx = random.randint(0, num_points)
        sample_idx[0] = selected_idx
        
        for i in range(num_sample):
            distance = point_coord[batch, selected_idx, :] - point_coord[batch, :num_points, :]
            distance = (distance**2).sum(-1)
            
            closest_distances = torch.min(distance, closest_distances)
            
            selected_idx = torch.argmax(closest_distances)
            sample_idx[i] = selected_idx
        
        sample_indices.append(sample_idx)
    sample_indices = torch.stack(sample_indices)
    
    return sample_indices

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
    