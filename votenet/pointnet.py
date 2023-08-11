import torch

from torch import nn

class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        
    def forward(self, x):
        ...
        
class TNet(nn.Module):
    def __init__(self, in_size):
        super().__init__()
        self.in_size = in_size
        self.out_sizes = [64, 128, 1024, 512, 256]
        
        self.mlp1 = nn.Sequential(
            nn.Conv2d(1, self.out_sizes[0], (1, in_size)),
            nn.BatchNorm2d(self.out_sizes[0]),
            nn.ReLU()
        ) # [B, 1, N, in_size] -> [B, 64, N, 1]
        self.mlp2 = nn.Sequential(
            nn.Conv2d(self.out_sizes[0], self.out_sizes[1], (1, 1)),
            nn.BatchNorm2d(self.out_sizes[1]),
            nn.ReLU()
        ) # [B, 64, N, 1] -> [B, 128, N, 1]
        self.mlp3 = nn.Sequential(
            nn.Conv2d(self.out_sizes[1], self.out_sizes[2], (1, 1)),
            nn.BatchNorm2d(self.out_sizes[2]),
            nn.ReLU()
        ) # [B, 128, N, 1] -> [B, 1024, N, 1]
        
        self.fc1 = nn.Sequential(
            nn.Linear(self.out_sizes[2], self.out_sizes[3]),
            nn.BatchNorm1d(self.out_sizes[3]),
            nn.ReLU()
        ) # [B, 1024] -> [B, 512]
        self.fc2 = nn.Linear(self.out_sizes[3], self.out_sizes[4]) # [B, 512] -> [B, 256]
                
        self.transform = nn.Linear(self.out_sizes[4], in_size * in_size)            
        nn.init.zeros_(self.transform.weight.data)
        self.transform.bias.data = torch.eye(in_size).view(in_size * in_size)
        
    def forward(self, x: torch.Tensor):
        # Input size: [B, N, in_size]
        num_points = x.shape[1]

        x = x.unsqueeze(1)
        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        
        x = nn.MaxPool2d((num_points, 1))(x)
        x = x.squeeze()
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        x = self.transform(x)
        x = x.view(-1, self.in_size, self.in_size)
        
        return x
    
if __name__ == '__main__':
    input_size = 3
    net = TNet(input_size)
    
    input = torch.randn((10, 1024, input_size))
    t = net(input)
    
    print(t.shape)
    print(t[0])