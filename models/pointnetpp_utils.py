import random
from typing import Sequence

import torch
from torch import Tensor

def farthestPointSampling(
    point_coord: Tensor,
    num_sample: int
) -> Tensor:
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

def gatherPoints(
    points: Tensor,
    indices: Tensor
):
    """
    Slicing point cloud tensors according to given indices.
    
    Args:
        points: (B, N, C) tensor
        indices: (B, N') tensor containing target indices
    Returns:
        new_points: (B, N', C) tensor
    """
    new_points_list = []
    for b, indices_batch in enumerate(indices):
        new_points_list_batch = [points[b, index, :] for index in indices_batch]
        new_points_batch = torch.stack(new_points_list_batch)
        
        new_points_list.append(new_points_batch)
    
    new_points = torch.stack(new_points_list)
    return new_points

def gatherPointsGrad(
    grad_output: Tensor,
    indices: Tensor,
    shape: Sequence[int]
):
    """
    Args:
        grad_output: (B, N', C) tensor of upstream gradients
        indices: (B, N') tensor containg target indices for forward calculation
        shape: (3) sequnece containing original tensor size components
    Returns:
        grad_features: (B, N, C) tensor of reconstructed downstream gradients
    """
    num_batch = shape[0]
    grad_features = torch.zeros(shape)
    
    for b in range(num_batch):
        for i, index in enumerate(indices[b]):
            grad_features[b, index] = grad_output[b, i]
    
    return grad_features

def ballQuery(
    point_coord: Tensor,
    centroid_coord: Tensor,
    radius: float,
    max_num_cluster: int
) -> Tensor:
    """
    Ball query clustering.
    
    Args:
        point_coord: (B, N, 3) tensor
        centroid_coord: (B, N', 3) tensor
        radius: radius of the ball
        max_num_cluster: maximum number of points in the ball
    Returns:
        cluster_point_indices: (B, N', num_sample) tensor containing indicies of each cluster
    """
    def getDistance(centroid_coord, point_coord):
        dist = centroid_coord - point_coord
        dist2 = dist ** 2
        dist2 = dist2.sum(1)
        return dist2

    radius2 = radius ** 2
    num_batch = centroid_coord.shape[0]
    num_centroid = centroid_coord.shape[1]
    
    cluster_point_indices_list = []
    for b in range(num_batch):
        indices_list_batch = []
        for i in range(num_centroid):
            dist2 = getDistance(centroid_coord[b, i], point_coord[b])
            indices = torch.zeros((max_num_cluster, ))
            cnt = 0
            for k, bool in enumerate(dist2 < radius2):
                if bool:
                    indices[cnt] = k
                    cnt += 1
                if cnt == max_num_cluster:
                    break
            
            indices_list_batch.append(indices)
        indices_batch = torch.stack(indices_list_batch)
        cluster_point_indices_list.append(indices_batch)
    
    cluster_point_indices = torch.stack(cluster_point_indices_list)
    return cluster_point_indices
