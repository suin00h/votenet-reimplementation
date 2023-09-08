import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

from pointnetpp_utils import (
    farthestPointSampling,
    gatherPoints,
    gatherPointsGrad,
    ballQuery
)

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
        num_sample: the number of points to be sampled by sampler
        radius: radius of ball query
        max_num_cluster: maximum number of cluster elements
    """
    def __init__(
        self,
        num_sample: int,
        radius: float,
        max_num_cluster: int
    ):
        super().__init__()
        
        self.num_sample = num_sample
        self.radius = radius
        self.max_num_cluster = max_num_cluster
        
        self.sampling = FarthestPointSampling.apply
        self.gather = GatherPoints.apply
        self.grouping = BallQuery.apply

    def forward(
        self,
        point_coord: Tensor,
        features: Tensor
    ):
        """
        Args:
            point_coord: (B, N, 3) tensor
            features: (B, N, C) tensor
        
        Returns:
            point_coord: (B, N, 3) tensor containing point clouds' xyz coordinates
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
        
        grouped_points = self.grouping(point_coord, centroid_coord, self.radius, self.max_num_cluster)
        
        return grouped_points

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
        point_coord: Tensor,
        num_sample: int
    ) -> Tensor:
        sample_indices = farthestPointSampling(point_coord, num_sample)
        ctx.mark_non_differentiable(sample_indices)
        
        return sample_indices
    
    @staticmethod
    def backward(ctx, grad_output=None):
        return None, None

class GatherPoints(Function):
    """
    Custom function for slicing point cloud tensors according to given indices.
    Wrapping gatherPoints and gatherPointsGrad function.
    
    Args:
        points: (B, N, C) tensor
        indices: (B, N') tensor containing target indices
    Returns:
        new_points: (B, N', C) tensor
    """
    @staticmethod
    def forward(
        ctx,
        points: Tensor,
        indices: Tensor
    ):
        ctx.for_backwards = (indices, points.shape)
        
        new_points = gatherPoints(points, indices)
        return new_points
    
    @staticmethod
    def backward(ctx, grad_output):
        indices, shape = ctx.for_backwards
        
        grad_features = gatherPointsGrad(grad_output, indices, shape)
        return grad_features, None

class BallQuery(Function):
    """
    Custom function for ball query algorithm implementation.
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

if __name__ == "__main__":
    sample_input = torch.randn((10, 100, 3), requires_grad=True)
    sa = SetAbstractionLayer(10, 0.2, 5)
    output = sa(sample_input, None)
    print(output.shape)
    print(output.is_contiguous())
    
    # from torch import optim
    # opt = optim.Adam(sa.parameters())
    # target = torch.randn((10, 10, 3))
    # loss = (target - output).view(-1).sum()
    # loss.backward()
    # opt.step()
