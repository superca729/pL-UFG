from typing import Optional, Tuple

from torch._C import BenchmarkExecutionStats
from torch_geometric.typing import Adj, OptTensor, PairTensor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul

from torch_geometric.utils import num_nodes
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils import get_laplacian

from torch_geometric.nn.inits import glorot, zeros
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import math

# function for pre-processing
@torch.no_grad()
def scipy_to_torch_sparse(A):
    A = sparse.coo_matrix(A)
    row = torch.tensor(A.row)
    col = torch.tensor(A.col)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data)

    return torch.sparse_coo_tensor(index, value, A.shape)

def pgnn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=False, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t, deg_inv_sqrt

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

        return edge_index, edge_weight, deg_inv_sqrt



def calc_M(f, edge_index, edge_weight, deg_inv_sqrt, num_nodes, mu, p):
    if isinstance(edge_index, SparseTensor):
        row, col, edge_weight = edge_index.coo()
    else:
        row, col = edge_index[0], edge_index[1]

    ## calculate M
    graph_grad = torch.pow(edge_weight, 0.5).view(-1, 1) * (
                deg_inv_sqrt[row].view(-1, 1) * f[row] - deg_inv_sqrt[col].view(-1, 1) * f[col])
    graph_grad = torch.pow(torch.norm(graph_grad, dim=1), p - 2)
    M = edge_weight * graph_grad
    M.masked_fill_(M == float('inf'), 0)
    alpha = (deg_inv_sqrt.pow(2) * scatter_add(M, col, dim=0, dim_size=num_nodes) + (2 * mu) / p).pow(-1)
    beta = 4 * mu / p * alpha
    M_ = alpha[row] * deg_inv_sqrt[row] * M * deg_inv_sqrt[col]
    return M_, beta

# function for pre-processing
def ChebyshevApprox(f, n):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)
    return c


# function for pre-processing
def get_operator(L, DFilters, n, s, J, Lev):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]
    d_list = list()
    for i in range(r):
        for l in range(Lev):
            d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))
    for ii in range(Lev-1):
        d_list.pop()
    return d_list


class F_pGNNConv_2(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor, Tensor]]
    _cached_adj_t: Optional[Tuple[SparseTensor, Tensor]]

    def __init__(self,
                 num_nodes: int,
                 in_channels: int,
                 out_channels: int,
                 mu: float,
                 p: float,
                 K: int,
                 s: float,
                 n: int,
                 Lev: int,
                 DFilters: list,
                 improved: bool = False,
                 cached: bool = True,
                 method: int = 2,
                 warmup: int = 10,
                 add_self_loops: bool = False,
                 normalize: bool = True,
                 bias: bool = True,
                 return_M_: bool = False,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        #super(F_pGNNConv_2, self).__init__(**kwargs)
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mu = mu
        self.p = p
        self.K = K
        self.improved = improved
        self.Lev = Lev
        self.n = n
        self.s = s
        self.r = len(DFilters)
        self.DFilters = DFilters
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.method = method
        self.warmup = warmup

        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_d_list = None

        self.return_M_ = return_M_
        self.filter = nn.Parameter(torch.Tensor(((self.r - 1) * self.Lev + 1), num_nodes))
        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=bias)

        if return_M_:
            self.new_edge_attr = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        nn.init.uniform_(self.filter, 0.9, 1.1)
        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_d_list = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        
        num_nodes = x.size(self.node_dim)

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight, deg_inv_sqrt = pgnn_norm(  # yapf: disable
                        edge_index, edge_weight, num_nodes,
                        self.improved, self.add_self_loops)
                    #if self.cached:
                    #    self._cached_edge_index = (edge_index, edge_weight, deg_inv_sqrt)
                    self._cached_edge_index = (edge_index, edge_weight, deg_inv_sqrt)
                else:
                    edge_index, edge_weight, deg_inv_sqrt = cache[0], cache[1], cache[2]
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index, deg_inv_sqrt = pgnn_norm(  # yapf: disable
                        edge_index, edge_weight, num_nodes,
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = (edge_index, deg_inv_sqrt)
                else:
                    edge_index, deg_inv_sqrt = cache[0], cache[1]
            d_list = self._cached_d_list
            if d_list is None:
                num_nodes = maybe_num_nodes(edge_index, num_nodes)
                L = get_laplacian(edge_index, normalization='sym')
                L = sparse.coo_matrix((L[1].cpu().numpy(), (L[0][0, :].cpu().numpy(), L[0][1, :].cpu().numpy())),
                                      shape=(num_nodes, num_nodes))
                lobpcg_init = np.random.rand(num_nodes, 1)
                lambda_max, _ = lobpcg(L, lobpcg_init)
                lambda_max = lambda_max[0]
                #J = np.log(lambda_max / np.pi) / np.log(self.s) + self.Lev - 1  # dilation level to start the decomposition
                J = np.log(lambda_max / np.pi) / np.log(self.s)
                d_list = get_operator(L, self.DFilters, self.n, self.s, J, self.Lev)
                self._cached_d_list = d_list

        final_out = torch.zeros_like(x)
        if self.method == 0:
            out = torch.zeros_like(x)
            for j in range(len(d_list)):
                out += torch.sparse.mm(d_list[j], torch.unsqueeze(self.filter[j], 1) * torch.sparse.mm(d_list[j], x))
            saved_out = out
            with torch.no_grad():
                for i in range(self.warmup):
                    edge_attr, beta = calc_M(out, edge_index, edge_weight, deg_inv_sqrt, num_nodes, self.mu, self.p)
                    out = self.propagate(edge_index, x=out, edge_weight=edge_attr, size=None) + beta.view(-1,1) * saved_out
            for _ in range(self.K):
                edge_attr, beta = calc_M(out, edge_index, edge_weight, deg_inv_sqrt, num_nodes, self.mu, self.p)
                out = self.propagate(edge_index, x=out, edge_weight=edge_attr, size=None) + beta.view(-1, 1) * saved_out
            final_out += out

        elif self.method == 1:
            ## New Model
            for j in range(len(d_list)):
                out = torch.sparse.mm(d_list[j], torch.unsqueeze(self.filter[j], 1) * torch.sparse.mm(d_list[j], x))
                saved_out = out
                with torch.no_grad():
                    for i in range(self.warmup):
                        edge_attr, beta = calc_M(out, edge_index, edge_weight, deg_inv_sqrt, num_nodes, self.mu, self.p)
                        out = self.propagate(edge_index, x=out, edge_weight=edge_attr, size=None) + beta.view(-1, 1) * saved_out  # x
                for _ in range(self.K):
                    edge_attr, beta = calc_M(out, edge_index, edge_weight, deg_inv_sqrt, num_nodes, self.mu, self.p)
                    out = self.propagate(edge_index, x=out, edge_weight=edge_attr, size=None) + beta.view(-1,1) * saved_out  # x
                final_out += out  #torch.sparse.mm(d_list[j], out)

        else:   # Method 2
            for j in range(len(d_list)):
                out = torch.unsqueeze(self.filter[j], 1) * torch.sparse.mm(d_list[j], x)
                saved_out = out
                with torch.no_grad():
                    for i in range(self.warmup):
                        edge_attr, beta = calc_M(out, edge_index, edge_weight, deg_inv_sqrt, num_nodes, self.mu, self.p)
                        out = self.propagate(edge_index, x=out, edge_weight=edge_attr, size=None) + beta.view(-1, 1) * saved_out #x
                for _ in range(self.K):
                    edge_attr, beta = calc_M(out, edge_index, edge_weight, deg_inv_sqrt, num_nodes, self.mu, self.p)
                    out = self.propagate(edge_index, x=out, edge_weight=edge_attr, size=None) + beta.view(-1, 1) * saved_out #x
                final_out += torch.sparse.mm(d_list[j], out)

        final_out = self.lin1(final_out)

        if self.return_M_:
            self.new_edge_attr = edge_attr

        return final_out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)

class UFGConv(nn.Module):

    def __init__(self,
                 num_nodes: int,
                 in_channels: int,
                 out_channels: int,
                 s: float,
                 n: int,
                 Lev: int,
                 DFilters: list,
                 improved: bool = False,
                 cached: bool = True,
                 method: int = 2,
                 warmup: int = 10,
                 add_self_loops: bool = False,
                 normalize: bool = True,
                 bias: bool = True,
                 return_M_: bool = False,
                 **kwargs):

        kwargs.setdefault('aggr', 'add')
        #super(F_pGNNConv_2, self).__init__(**kwargs)
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.Lev = Lev
        self.n = n
        self.s = s
        self.r = len(DFilters)
        self.DFilters = DFilters
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.method = method
        self.warmup = warmup

        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_d_list = None

        self.return_M_ = return_M_
        self.filter = nn.Parameter(torch.Tensor(((self.r - 1) * self.Lev + 1), num_nodes))
        self.lin1 = torch.nn.Linear(in_channels, out_channels, bias=bias)

        if return_M_:
            self.new_edge_attr = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        nn.init.uniform_(self.filter, 0.9, 1.1)
        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_d_list = None
        
        
    def reset_parameters(self):
        self.lin1.reset_parameters()
        nn.init.uniform_(self.filter, 0.9, 1.1)
        self._cached_edge_index = None
        self._cached_adj_t = None
        self._cached_d_list = None

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        """"""
        num_nodes = x.size(-2)

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight, deg_inv_sqrt = pgnn_norm(  # yapf: disable
                        edge_index, edge_weight, num_nodes,
                        self.improved, self.add_self_loops)
                    #if self.cached:
                    #    self._cached_edge_index = (edge_index, edge_weight, deg_inv_sqrt)
                    self._cached_edge_index = (edge_index, edge_weight, deg_inv_sqrt)
                else:
                    edge_index, edge_weight, deg_inv_sqrt = cache[0], cache[1], cache[2]
            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index, deg_inv_sqrt = pgnn_norm(  # yapf: disable
                        edge_index, edge_weight, num_nodes,
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = (edge_index, deg_inv_sqrt)
                else:
                    edge_index, deg_inv_sqrt = cache[0], cache[1]
            d_list = self._cached_d_list
            if d_list is None:
                num_nodes = maybe_num_nodes(edge_index, num_nodes)
                L = get_laplacian(edge_index, normalization='sym')
                L = sparse.coo_matrix((L[1].cpu().numpy(), (L[0][0, :].cpu().numpy(), L[0][1, :].cpu().numpy())),
                                      shape=(num_nodes, num_nodes))
                lobpcg_init = np.random.rand(num_nodes, 1)
                lambda_max, _ = lobpcg(L, lobpcg_init)
                lambda_max = lambda_max[0]
                #J = np.log(lambda_max / np.pi) / np.log(self.s) + self.Lev - 1  # dilation level to start the decomposition
                J = np.log(lambda_max / np.pi) / np.log(self.s)
                d_list = get_operator(L, self.DFilters, self.n, self.s, J, self.Lev)
                self._cached_d_list = d_list
                
        final_out = torch.zeros_like(x)
        if self.method == 0:
            x = torch.sparse.mm(torch.cat(d_list, dim=0), x)
            x = self.filter.reshape((self.r * self.Lev * num_nodes,1)) * x
            x = torch.sparse.mm(torch.cat(d_list, dim=1), x)
            final_out = x
            # with torch.no_grad():
            #     for i in range(self.warmup):
            #         edge_attr, beta = calc_M(out, edge_index, edge_weight, deg_inv_sqrt, num_nodes, self.mu, self.p)
            #         out = self.propagate(edge_index, x=out, edge_weight=edge_attr, size=None) + beta.view(-1,1) * saved_out
            # for _ in range(self.K):
            #     edge_attr, beta = calc_M(out, edge_index, edge_weight, deg_inv_sqrt, num_nodes, self.mu, self.p)
            #     out = self.propagate(edge_index, x=out, edge_weight=edge_attr, size=None) + beta.view(-1, 1) * saved_out
            # final_out += out
                
            return final_out


            

        
        
    
