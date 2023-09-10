import torch.nn as nn
from torch import Tensor

from pointnetpp_utils import (
    FarthestPointSampling,
    GatherPoints,
    GroupingLayer
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
        
        self.sampling = FarthestPointSampling.apply
        self.gather = GatherPoints.apply
        self.grouping = GroupingLayer(radius, max_num_cluster)

    def forward(
        self,
        point_coord: Tensor,
        features: Tensor
    ):
        """
        Args:
            point_coord: (B, N, 3) tensor
            features: (B, C, N) tensor
        
        Returns:
            point_coord: (B, N, 3) tensor containing point clouds' xyz coordinates
            features: (B, C, N) tensor
        
        Todo:
            Sampling layer: use farthest point sampling(FPS) to get subset of points.
            Grouping layer: use ball query algorithm to get N' clusters of points.
                Each clusters have different number of points upper-limited to K.
            PointNet layer: each clusters are processed within PointNet-like module
                and the output is abstracted by its centroid and local feature.
        """
        centroid_indices = self.sampling(point_coord, self.num_sample)
        centroid_coord = self.gather(point_coord, centroid_indices)
        
        grouped_features = self.grouping(point_coord, centroid_coord, features)
        
        return grouped_features

if __name__ == "__main__":
    import torch
    
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
