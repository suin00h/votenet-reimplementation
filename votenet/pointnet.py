import torch

from torch import nn

class PointNet(nn.Module):
    def __init__(self, enable_tnet=True, require_global_feature=True, class_num: int=40):
        '''
            enable_tnet: [True|False] Determine whether to use t-net inside the model.
            require_global_feature: [True|False] If true, the model outputs just a global feature vector of size [B, 1024], else returns class scores.
        '''
        super().__init__()
        
        self.enable_tnet = enable_tnet
        self.require_global_feature = require_global_feature
        self.channels = [1, 64, 64, 64, 128, 1024, 512, 256, class_num]
        
        self.input_transform = TNet(size=3, in_channel=1, first_kernel_size=(1, 3))
        self.feature_transform = TNet(size=64, in_channel=64)
        
        self.mlp = [self.get_mlp(self.channels[0], self.channels[1], (1, 3))]
        self.mlp += [self.get_mlp(self.channels[i], self.channels[i+1]) for i in range(1, 5)]
        
        self.fc1 = nn.Sequential(
            nn.Linear(self.channels[5], self.channels[6]),
            nn.BatchNorm1d(self.channels[6]),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(self.channels[6], self.channels[7]),
            nn.BatchNorm1d(self.channels[7]),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        self.fc3 = nn.Linear(self.channels[7], self.channels[8])
    
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
        
        if self.require_global_feature:
            return x
        
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        x = nn.Softmax(dim=1)(x)
        return x
        
class TNet(nn.Module):
    def __init__(self, size, in_channel, first_kernel_size=(1, 1)):
        super().__init__()
        self.size = size
        self.channels = [in_channel, 64, 128, 1024, 512, 256]
        
        self.mlp = [self.get_mlp(self.channels[0], self.channels[1], first_kernel_size)]
        self.mlp += [self.get_mlp(self.channels[i], self.channels[i+1]) for i in range(1, 3)]
        
        self.fc1 = nn.Sequential(
            nn.Linear(self.channels[3], self.channels[4]),
            nn.BatchNorm1d(self.channels[4]),
            nn.ReLU()
        ) # [B, 1024] -> [B, 512]
        self.fc2 = nn.Linear(self.channels[4], self.channels[5]) # [B, 512] -> [B, 256]
                
        self.transform = nn.Linear(self.channels[5], size * size)            
        nn.init.zeros_(self.transform.weight.data)
        self.transform.bias.data = torch.eye(size).view(size * size)
        
    def get_mlp(self, in_channel, out_channel, kernel_size=(1, 1)):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size),
            nn.BatchNorm2d(out_channel),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor):
        num_points = x.shape[2]

        x = self.mlp[0](x)
        x = self.mlp[1](x)
        x = self.mlp[2](x)
        
        x = nn.MaxPool2d((num_points, 1))(x)
        x = x.squeeze()
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        x = self.transform(x)
        x = x.view(-1, self.size, self.size)
        
        return x
    
if __name__ == '__main__':
    with torch.no_grad():
        print('Test batch size: 10\n')
        # Test input transform
        net = TNet(size=3, in_channel=1, first_kernel_size=(1, 3))
        input = torch.randn((10, 1, 1024, 3))
        t = net(input)
        
        print('Input transform matrix shape:', t.shape)
        print('Matrix sample:\n', t[0], sep='', end='\n\n')
        
        # Test feature transform
        net = TNet(size=64, in_channel=64)
        
        input = torch.randn((10, 64, 1024, 1))
        t = net(input)
        
        print('Feature transform matrix shape:', t.shape, end='\n\n')
        
        # Test PointNet global feature
        net = PointNet()
        
        input = torch.randn((10, 1024, 3))
        global_feature = net(input)
        
        print('Global feature shape:', global_feature.shape)
        
        # Test PointNet class scores
        net = PointNet(require_global_feature=False)
        
        cls_scores = net(input)
        
        print('Class score shape:', cls_scores.shape)