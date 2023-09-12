import random
from typing import Sequence

import torch
from torch import nn
from torch import Tensor
from torch.autograd import Function

class FarthestPointSampling(Function):
    """
    Custom function implementation of farthest point sampling(FPS).
    Wrapping farthestPointSampling function.
    
    Args:
        point_coord: (B, N, 3) tensor
        num_sample: number of points to be sampled
    Returns:
        sample_idxs: (B, num_sample) tensor of indices
    """
    @staticmethod
    def forward(
        ctx,
        point_coord: Tensor,
        num_sample: int
    ) -> Tensor:
        sample_idxs = farthestPointSampling(point_coord, num_sample)
        ctx.mark_non_differentiable(sample_idxs)
        return sample_idxs
    
    @staticmethod
    def backward(ctx, grad_output=None):
        return None, None

def farthestPointSampling(
    point_coord: Tensor,
    num_sample: int
) -> Tensor:
    """
    Finds indices of a subset of points that are the farthest away from each other.
    Source code courtesy: pytorch3d docs
    
    Args:
        point_coord: (B, N, 3) tensor
        num_sample: number of points to be sampled
    Returns:
        sample_idxs: (B, num_sample) tensor of indices
    """
    num_batch, num_points, _ = point_coord.shape
    device = point_coord.device
    
    sample_idxs = []    
    for b in range(num_batch):
        sample_idx = torch.full((num_sample, ), -1, dtype=torch.int64, device=device)
        closest_dist = torch.full((num_points, ), float("inf"), dtype=torch.float32, device=device)
        selected_idx = random.randint(0, num_points - 1)
        
        sample_idx[0] = selected_idx
        for i in range(num_sample):
            dist = point_coord[b, selected_idx, :] - point_coord[b, :num_points, :]
            dist = (dist ** 2).sum(-1)
            
            closest_dist = torch.min(dist, closest_dist)
            
            selected_idx = torch.argmax(closest_dist)
            sample_idx[i] = selected_idx
        
        sample_idxs.append(sample_idx)
    
    sample_idxs = torch.stack(sample_idxs)
    return sample_idxs

class GatherPoints(Function):
    """
    Custom function for slicing point cloud tensors according to given indices.
    Wrapping gatherPoints and gatherPointsGrad function.
    
    Args:
        points: (B, N, C) tensor
        idxs: (B, N') tensor containing target indices
    Returns:
        new_points: (B, N', C) tensor
    """
    @staticmethod
    def forward(
        ctx,
        points: Tensor,
        idxs: Tensor
    ):
        ctx.for_backwards = (idxs, points.size(1))
        
        new_points = gatherPoints(points, idxs)
        return new_points
    
    @staticmethod
    def backward(ctx, grad_output):
        idxs, num_points = ctx.for_backwards
        
        grad_input = gatherPointsGrad(grad_output, idxs, num_points)
        return grad_input, None

def gatherPoints(
    points: Tensor,
    idxs: Tensor
):
    """
    Slicing point cloud tensors according to given indices.
    
    Args:
        points: (B, N, C) tensor
        idxs: (B, N') tensor containing target indices
    Returns:
        new_points: (B, N', C) tensor
    """
    new_points = torch.zeros(
        (points.size(0), idxs.size(1), points.size(2)),
        dtype=torch.float32, device=points.device
    )
    
    for b in range(points.size(0)):
        for i, idx in enumerate(idxs[b]):
            new_points[b, i] = points[b, idx]
    return new_points

def gatherPointsGrad(
    grad_output: Tensor,
    idxs: Tensor,
    num_points: int
):
    """
    Args:
        grad_output: (B, N', C) tensor of upstream gradients
        idxs: (B, N') tensor containg target indices for forward calculation
        num_points: number of original points
    Returns:
        grad_input: (B, N, C) tensor of reconstructed downstream gradients
    """
    grad_input = torch.zeros(
        (grad_output.size(0), num_points, grad_output.size(2)),
        dtype=torch.float32, device=grad_output.device
    )
    
    for b in range(grad_output.size(0)):
        for i, idx in enumerate(idxs[b]):
            grad_input[b, idx] += grad_output[b, i]
    return grad_input

class BallQuery(Function):
    """
    Custom function implementation of ball query algorithm.
    Wrapping ballQuery function.
    
    Args:
        point_coord: (B, N, 3) tensor
        centroid_coord: (B, N', 3) tensor
        radius: radius of the ball
        max_num_cluster: maximum number of points in the ball
    Returns:
        cluster_point_indices: (B, N', num_sample) tensor containing indicies of each cluster
    """
    @staticmethod
    def forward(
        ctx,
        point_coord: Tensor,
        centroid_coord: Tensor,
        radius: float,
        max_num_cluster: int
    ):
        cluster_point_indices = ballQuery(point_coord, centroid_coord, radius, max_num_cluster)
        ctx.mark_non_differentiable(cluster_point_indices)
        
        return cluster_point_indices
    
    @staticmethod
    def backward(ctx, grad_output):
        return None, None, None, None

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

class GroupingLayer(nn.Module):
    """
    Grouping point features with a ball query.
    
    Args:
        radius: radius of the ball
        max_num_cluster: maximum number of points in the ball
        use_coord: whether to use coordinate as a feature
        normalize_coord: whether to normalize coordinates
    """
    def __init__(
        self,
        radius: float,
        max_num_cluster: int,
        use_coord: bool=True,
        normalize_coord: bool=True
    ):
        super().__init__()
        
        self.radius = radius
        self.max_num_cluster = max_num_cluster
        self.use_coord = use_coord
        self.normalize_coord = normalize_coord
        
        self.ball_query = BallQuery.apply
        self.grouper = ...
    
    def forward(
        self,
        point_coord: Tensor,
        centroid_coord: Tensor,
        features: Tensor=None
    ):
        """
        Args:
            point_coord: (B, N, 3) tensor
            centroid_coord: (B, N', 3) tensor
            features: (B, C, N) tensor
        
        Returns:
            grouped_features: (B, C+3, N', K)
        """
        cluster_indices = self.ball_query(point_coord, centroid_coord, self.radius, self.max_num_cluster)
        
        grouped_coord = grouped_coord.transpose(1, 2).contiguous()
        grouped_coord = self.grouper(point_coord, cluster_indices)
        grouped_coord -= centroid_coord.transpose(1, 2).unsqueeze(-1)
        if self.normalize_coord:
            grouped_coord /= self.radius
            
        if features is None:
            return grouped_coord
        grouped_features = self.grouper(features, cluster_indices)
        if self.use_coord:
            grouped_features = torch.cat((grouped_coord, grouped_features), dim=1)
            
        return grouped_features

if __name__ == "__main__":
    from torch.autograd import gradcheck
    B, N, C = (2, 3, 2)
    x = torch.randn((B, N, 2), dtype=torch.float32, requires_grad=True)
    y = FarthestPointSampling.apply(x, 2)
    print(x, y)
    print(gradcheck(GatherPoints.apply, (x, y)))
    # print(gradcheck(FarthestPointSampling.apply, (x, 3)))
