import os.path as osp
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WebKB, Actor, WikipediaNetwork, Amazon, Coauthor, CoraFull
from torch_geometric.utils import num_nodes
from cSBM_dataset import dataset_ContextualSBM
from torch_geometric.utils import to_undirected
from torch_geometric.utils import get_laplacian
from scipy import sparse

# def nodeAttack(x, ratio_extreme, extreme_value,ratio=0.3,std=0.1, normal=False):
#     if normal:
#             x_new= x + torch.normal(0, std, size=(x.shape[0], x.shape[1])).to(device)
#             mask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() < ratio_extreme
#             mask = mask.to(device)
#             polarity_mask = (torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() < 0.5).to(device)
#             polarity_mask = polarity_mask * 2
#             polarity_mask = polarity_mask - 1
#             x_new[mask]=x_new[mask]+extreme_value*polarity_mask[mask]
#             #x_new=x_new.clamp(min=0)

#     else:
#             mask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() < ratio
#             mask = mask.to(device)
#             x -= mask.int()
#             x_new = (torch.abs(x)==1).double()
#             mask_2 = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() < ratio_extreme
#             mask_2 = mask_2.to(device)
#             polarity_mask = torch.rand(x.shape).to(device)
#             # polarity_mask = polarity_mask * 2
#             # polarity_mask = polarity_mask - 1
#             x_new[mask_2] = extreme_value*polarity_mask[mask_2]

#     return x_new.to(device)

def add_noise(x, ratio):
    mask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() < ratio
    mask = mask
    x -= mask.int()
    x_new = torch.abs(x)
    
    return x_new
    
    
def generate_split(data, num_classes, seed=2021, train_num_per_c=20, val_num_per_c=30):
    np.random.seed(seed)
    train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    for c in range(num_classes):
        all_c_idx = (data.y == c).nonzero()
        if all_c_idx.size(0) <= train_num_per_c + val_num_per_c:
            test_mask[all_c_idx] = True
            continue
        perm = np.random.permutation(all_c_idx.size(0))
        c_train_idx = all_c_idx[perm[:train_num_per_c]]
        train_mask[c_train_idx] = True
        test_mask[c_train_idx] = True
        c_val_idx = all_c_idx[perm[train_num_per_c : train_num_per_c + val_num_per_c]]
        val_mask[c_val_idx] = True
        test_mask[c_val_idx] = True
    test_mask = ~test_mask

    return train_mask, val_mask, test_mask

def generate_random_edges(data, random_rate=None, seed=2021):
    np.random.seed(seed)
    n_nodes = num_nodes.maybe_num_nodes(data.edge_index)
    if random_rate == 0:
        return data.edge_index
    elif random_rate == 1:
        num_new_edges = len(data.edge_index.T)
        rd_edge_index_1 = np.random.randint(n_nodes-1, size=(2, num_new_edges))
        rd_edge_index_2 = np.random.randint(n_nodes-1, size=(2, num_new_edges))
        new_edge_index = np.concatenate([rd_edge_index_1.T, rd_edge_index_2.T])
        new_edge_index = list(set([tuple(e_index) for e_index in new_edge_index]))
        new_edge_index = [list(v) for v in new_edge_index]
        new_edge_index = new_edge_index[:num_new_edges]
        new_edge_index.sort()
        new_edge_index = torch.LongTensor(new_edge_index)
        return new_edge_index.T
    else:
        num_new_edges = int(random_rate * len(data.edge_index.T))
        rd_edge_index = np.random.randint(n_nodes-1, size=(2, num_new_edges))
        old_edge_index = data.edge_index.numpy().T
        rm_id = np.random.choice(len(data.edge_index.T)-1, num_new_edges)
        old_edge_index = np.delete(old_edge_index, rm_id, 0)

        new_edge_index = np.concatenate([old_edge_index, rd_edge_index.T])
        new_edge_index = list(set([tuple(e_index) for e_index in new_edge_index]))
        new_edge_index = [list(v) for v in new_edge_index]
        new_edge_index.sort()
        
        new_edge_index = torch.LongTensor(new_edge_index)
        return new_edge_index.T

    

def load_data(args, root='dataset', rand_seed=2021):
    dataset = args.input
    path = osp.join(root, dataset)
    dataset = dataset.lower()

    if 'csbm' in dataset:
        dataset = dataset_ContextualSBM(root, 
                                        name=dataset, 
                                        train_percent=args.train_rate, 
                                        val_percent=args.val_rate)
        num_features = dataset.num_features
        num_classes = dataset.num_classes
        data = dataset[0]
        return data, num_features, num_classes
    elif dataset in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(path, dataset)
    elif dataset in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, dataset)
    elif dataset == 'actor':
        dataset = Actor(path)
    elif dataset in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(path, dataset)
    elif dataset in ['computers', 'photo']:
        dataset = Amazon(path, dataset)
    elif dataset in ['cs', 'physics']:
        dataset = Coauthor(path, dataset)

    num_features = dataset.num_features
    num_classes = dataset.num_classes
    data = dataset[0]
    #noises = torch.tensor(np.random.normal(args.mu_noise, args.sigma_noise, size = (data.x.shape[0], data.x.shape[1]))).float() # add by jessica
    #data.x = data.x + noises #add  by jessica (noise)
    data.x = add_noise(data.x, args.ratio)
    #data.edge_index = to_undirected(data.edge_index, data.num_nodes) #add by jessica
    num_train = int(len(data.y) / num_classes * args.train_rate)
    num_val = int(len(data.y) / num_classes * args.val_rate)
    data.train_mask, data.val_mask, data.test_mask = generate_split(data, num_classes, rand_seed, num_train, num_val)

    return data, num_features, num_classes