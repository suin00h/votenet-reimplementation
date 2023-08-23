import random

import torch
import torch.nn as nn

class PointNetpp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class SetAbstraction(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.set_abstraction_layers = nn.ModuleList([])

    def forward(self, x):
        return x
    
class SetAbstractionLayer(nn.Module):
    '''
        Takes an [B, N, (d + C)] tensor as input, that is from N points with d-dim coordinates and C-dim point feature.
        Outputs an [B, N', (d + C')] tensor of N' subsampled points with d-dim coordinates and C'-dim point feature.
    '''
    def __init__(
        self,
        num_sample: int,
        ball_query_radius: float
    ):
        super().__init__()
        
        self.num_sample = num_sample
        self.ball_query_radius = ball_query_radius

    def forward(self, x):
        point_clouds = x[:,:, :3]
        # [1] Sampling layer
        centroid_indices_batch = FPS(points=point_clouds, num_sample=self.num_sample)
        
        for point_cloud, centroid_indices in zip(point_clouds, centroid_indices_batch):
            # [2] Grouping Layer
            cluster_list = ball_query(points=point_cloud, centroid_indices=centroid_indices, radius=self.ball_query_radius)
            
            # [3] PointNet Layer
        
        return x
    
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
    