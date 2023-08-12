import torch
import torch.nn.functional as F
import warnings
warnings.filterwarnings(action='ignore')

from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader
from dataset.dataset import MN40
from votenet.pointnet import PointNet
from utils.data import get_cls_idx

def train(
    net: PointNet,
    dataloader: DataLoader,
    opt: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: torch.device
):
    opt.zero_grad()
    net.train()
    
    for item in tqdm(dataloader):
        input, cls = item['cloud'].to(device), F.one_hot(item['cls'], num_classes=40).type(torch.FloatTensor).to(device)
        
        output = net(input)
        loss = loss_fn(output, cls)
        loss.backward()
        opt.step()
        

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = PointNet(enable_tnet=True, require_global_feature=False).to(device)
train_dataset = MN40(split='train')

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

opt = optim.Adam(net.parameters())
loss_fn = torch.nn.CrossEntropyLoss()

for epoch in range(1):
    train(net=net, dataloader=train_dataloader, opt=opt, loss_fn=loss_fn, device=device)