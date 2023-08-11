import torch

from torch import nn

class PointNet(nn.Module):
    def __init__(self, enable_tnet=True):
        super().__init__()
        
        self.enable_tnet = enable_tnet
        self.channels = [1, 64, 64, 64, 128, 1024]
        
        self.input_transform = TNet(size=3, in_channel=1, first_kernel_size=(1, 3))
        self.feature_transform = TNet(size=64, in_channel=64, first_kernel_size=(1, 1))
        
        self.mlp = [self.get_mlp(self.channels[0], self.channels[1], (1, 3))] + [self.get_mlp(self.channels[i], self.channels[i+1]) for i in range(1, 5)]
    
    def get_mlp(self, in_channel, out_channel, kernel_size=(1, 1)):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor):
        num_points = x.shape[1]
        
        x = x.unsqueeze(1) # [B, 1, N, 3]
        input_transform = self.input_transform(x).unsqueeze(1) if self.enable_tnet else torch.eye(3)
        x = torch.matmul(x, input_transform)
        
        x = self.mlp[0](x)
        x = self.mlp[1](x) # [B, 64, N, 1]
        
        feature_transform = self.feature_transform(x) if self.enable_tnet else torch.eye(64)
        x = torch.matmul(feature_transform, x.squeeze())
        
        x = x.unsqueeze(3)
        x = self.mlp[2](x)
        x = self.mlp[3](x)
        x = self.mlp[4](x)
        
        x = nn.MaxPool2d((num_points, 1))(x)
        x = x.squeeze()
        
        return x
        
class TNet(nn.Module):
    def __init__(self, size, in_channel, first_kernel_size=(1, 1)):
        super().__init__()
        self.size = size
        self.out_channels = [64, 128, 1024, 512, 256]
        
        self.mlp1 = nn.Sequential(
            nn.Conv2d(in_channel, self.out_channels[0], first_kernel_size),
            nn.BatchNorm2d(self.out_channels[0]),
            nn.ReLU()
        ) # [B, in_channel, N, in_size] -> [B, 64, N, 1]
        self.mlp2 = nn.Sequential(
            nn.Conv2d(self.out_channels[0], self.out_channels[1], (1, 1)),
            nn.BatchNorm2d(self.out_channels[1]),
            nn.ReLU()
        ) # [B, 64, N, 1] -> [B, 128, N, 1]
        self.mlp3 = nn.Sequential(
            nn.Conv2d(self.out_channels[1], self.out_channels[2], (1, 1)),
            nn.BatchNorm2d(self.out_channels[2]),
            nn.ReLU()
        ) # [B, 128, N, 1] -> [B, 1024, N, 1]
        
        self.fc1 = nn.Sequential(
            nn.Linear(self.out_channels[2], self.out_channels[3]),
            nn.BatchNorm1d(self.out_channels[3]),
            nn.ReLU()
        ) # [B, 1024] -> [B, 512]
        self.fc2 = nn.Linear(self.out_channels[3], self.out_channels[4]) # [B, 512] -> [B, 256]
                
        self.transform = nn.Linear(self.out_channels[4], size * size)            
        nn.init.zeros_(self.transform.weight.data)
        self.transform.bias.data = torch.eye(size).view(size * size)
        
    def forward(self, x: torch.Tensor):
        num_points = x.shape[2]

        x = self.mlp1(x)
        x = self.mlp2(x)
        x = self.mlp3(x)
        
        x = nn.MaxPool2d((num_points, 1))(x)
        x = x.squeeze()
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        x = self.transform(x)
        x = x.view(-1, self.size, self.size)
        
        return x
    
if __name__ == '__main__':
    with torch.no_grad():
        # Test input transform
        net = TNet(size=3, in_channel=1, first_kernel_size=(1, 3))
        input = torch.randn((10, 1, 1024, 3))
        t = net(input)
        
        print('Output shape:', t.shape)
        print(t[0], end='\n')
        
        # Test feature transform
        net = TNet(size=64, in_channel=64)
        
        input = torch.randn((10, 64, 1024, 1))
        t = net(input)
        
        print('Output shape:', t.shape, end='\n')
        
        # Test PointNet
        net = PointNet()
        
        input = torch.randn((10, 1024, 3))
        global_feature = net(input)
        
        print('Global feature shape:', global_feature.shape)