import torch
import torch.nn.functional as F
from src.framelet_conv import F_pGNNConv_2
from src.framelet_conv import UFGConv

class UFGNet(torch.nn.Module):
    def __init__(self,
                 num_nodes,
                 in_channels, 
                 out_channels,
                 num_hid,
                 #mu,
                 #p,
                 #K,
                 DFilters,
                 s,
                 n,
                 Lev,
                 dropout,
                 cached=True,
                 method = 0, warmup = 10):
        
        super(UFGNet, self).__init__()
        #super().all(*args, **kwargs)
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.conv1 = UFGConv(num_nodes, num_hid, out_channels, s, n, Lev, DFilters, cached=cached, method = method, warmup = warmup)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)        
        return F.log_softmax(x, dim=1)


class F_pGNNet_2(torch.nn.Module):
    def __init__(self,
                 num_nodes,
                 in_channels, 
                 out_channels,
                 num_hid,
                 mu,
                 p,
                 K,
                 DFilters,
                 s,
                 n,
                 Lev,
                 dropout,
                 cached=True,
                 method = 2, warmup = 10):
        super(F_pGNNet_2, self).__init__()
        self.dropout = dropout
        self.lin1 = torch.nn.Linear(in_channels, num_hid)
        self.conv1 = F_pGNNConv_2(num_nodes, num_hid, out_channels, mu, p, K, s, n, Lev, DFilters, cached=cached, method = method, warmup = warmup)

    def forward(self, x, edge_index, edge_weight=None):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index, edge_weight)        
        return F.log_softmax(x, dim=1)