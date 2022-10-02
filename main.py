import torch
import argparse
import time
import numpy as np
import os
import pandas as pd

from data_proc import load_data
from models import *
import torch_geometric.transforms as T
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# extract decomposition/reconstruction Masks
def getFilters(FrameType):
    if FrameType == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
    elif FrameType == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        DFilters = [D1, D2, D3]
    elif FrameType == 'Quadratic':  # not accurate so far
        D1 = lambda x: np.cos(x / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        D4 = lambda x: np.sin(x / 2) ** 3
        DFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')
    return DFilters

def build_model(args, num_features, num_classes, num_nodes = None):
    if args.model == 'pgnn':
        model = pGNNNet(in_channels=num_features,
                            out_channels=num_classes,
                            num_hid=args.num_hid,
                            mu=args.mu,
                            p=args.p,
                            K=args.K,
                            dropout=args.dropout)
    elif args.model == 'ufg_pgnn':
        DFilters = getFilters(args.FrameType)
        model = F_pGNNet_2(num_nodes = num_nodes,
                            in_channels=num_features,
                            out_channels=num_classes,
                            num_hid=args.num_hid,
                            mu=args.mu,
                            p=args.p,
                            K=args.K,
                            DFilters = DFilters,
                            s = args.s,
                            n = args.n,
                            Lev = args.Lev,
                            dropout=args.dropout, method = args.method, warmup = args.warmup)
    elif args.model == 'mlp':
        model = MLPNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'gcn':
        model = GCNNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        dropout=args.dropout)
    elif args.model == 'sgc':
        model = SGCNet(in_channels=num_features,
                        out_channels=num_classes,
                        K=args.K)
    elif args.model == 'gat':
        model = GATNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        num_heads=args.num_heads,
                        dropout=args.dropout)
    elif args.model == 'jk':
        model = JKNet(in_channels=num_features,
                        out_channels=num_classes,
                        num_hid=args.num_hid,
                        K=args.K,
                        alpha=args.alpha,
                        dropout=args.dropout)
    elif args.model == 'appnp':
        model = APPNPNet(in_channels=num_features,
                            out_channels=num_classes,
                            num_hid=args.num_hid,
                            K=args.K,
                            alpha=args.alpha,
                            dropout=args.dropout)
    elif args.model == 'gprgnn':
        model = GPRGNNNet(in_channels=num_features,
                            out_channels=num_classes,
                            num_hid=args.num_hid,
                            ppnp=args.ppnp,
                            K=args.K,
                            alpha=args.alpha,
                            Init=args.Init,
                            Gamma=args.Gamma,
                            dprate=args.dprate,
                            dropout=args.dropout)
    elif args.model == 'ufg':
        DFilters = getFilters(args.FrameType)
        model = UFGNet(num_nodes = num_nodes,
                            in_channels=num_features,
                            out_channels=num_classes,
                            num_hid=args.num_hid,
                            #mu=args.mu,
                            #p=args.p,
                            #K=args.K,
                            DFilters = DFilters,
                            s = args.s,
                            n = args.n,
                            Lev = args.Lev,
                            dropout=args.dropout, method = args.method, warmup = args.warmup)
    return model

def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    outputs = model(data.x, data.edge_index, data.edge_attr)
    loss = F.nll_loss(outputs[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


@torch.no_grad()
def test(model, data):
    model.eval()
    logits, accs = model(data.x, data.edge_index, data.edge_attr), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs

def main(args, name_list):
    print(args)

    ResultCSV = args.input+'_'+args.model + '_Exp.csv'

    data, num_features, num_classes = load_data(args, rand_seed=2021)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #results = []
    results = np.zeros(args.runs)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    for run in range(args.runs):
        model = build_model(args, num_features, num_classes, data.x.shape[0])
        model = model.to(device)
        data = data.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 
        
        t1 = time.time()
        best_val_acc = test_acc = 0
        for epoch in range(1, args.epochs+1):
            train(model, optimizer, data)
            train_acc, val_acc, tmp_test_acc = test(model, data)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))
        t2 = time.time()
        print('{}, {}, Accuacy: {:.4f}, Time: {:.4f}'.format(args.model, args.input, test_acc, t2-t1))
        #results.append(test_acc)
        results[run] = test_acc
    #results = 100 * torch.Tensor(results)
    results = 100.0 * results
    print(results)
    print(f'Averaged test accuracy for {args.runs} runs: {results.mean():.2f} \pm {results.std():.2f}')

    if os.path.isfile(ResultCSV):
        df = pd.read_csv(ResultCSV)
        ExpNum = df['ExpNum'].iloc[-1] + 1
    else:
        outputs_names = {'ExpNum': 'int'}
        ExpNum = 1
        #outputs_names.update({name: type(value).__name__ for (name, value) in args._get_kwargs()})
        outputs_names.update({name: value for (name, value) in name_list})
        outputs_names.update({'Replicate{0:2d}'.format(ii): 'float' for ii in range(1,args.runs+1)})
        outputs_names.update({'Ave_Test_Acc': 'float', 'Test_Acc_std': 'float'})
        df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in outputs_names.items()})

    #new_row = {name: value for (name, value) in args._get_kwargs()}
    new_row = {'ExpNum': ExpNum}
    new_row.update({name: value for (name, value) in args._get_kwargs()})
    #new_row.update({name: eval('args.' + name) for (name, value) in name_list[1:]})
    new_row.update({'Replicate{0:2d}'.format(ii): results[ii-1] for ii in range(1,args.runs+1)})
    new_row.update({'Ave_Test_Acc': np.mean(results), 'Test_Acc_std': np.std(results)})
    df = df.append(new_row, ignore_index=True)
    df.to_csv(ResultCSV, index=False)

def get_args():
    name_list = [('ExpNum', 'int')]
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='cora', help='Input graph.')  #['cora', 'citeseer', 'pubmed']['cornell', 'texas', 'wisconsin']
    name_list.append(('input', 'str')) #actor 'chameleon', 'squirrel' computers', 'photo' 'cs', 'physics'

    parser.add_argument('--train_rate', type=float, default=0.2, help='Training rate.')#025,  #0.6,   # 0.025,
    name_list.append(('train_rate', 'float'))

    parser.add_argument('--val_rate', type=float, default=0.1, help='Validation rate.') #0.025,  #0.2,   #0.025,
    name_list.append(('val_rate', 'float'))

    parser.add_argument('--model', type=str, default='ufg',
                        choices=['ufg','pgnn', 'ufg_pgnn', 'mlp', 'gcn', 'cheb', 'sgc', 'gat', 'jk', 'appnp', 'gprgnn'],
                        help='GNN model')
    name_list.append(('model', 'str'))
#%%    
    parser.add_argument('--mu', type=float, default=10, help='mu.') #0.1 0.5 1 5 10
    name_list.append(('mu', 'float'))

    parser.add_argument('--p', type=float, default=1, help='p.') 
    name_list.append(('p', 'float'))

    parser.add_argument('--Lev', type=int, default=1, help='level of transform (default: 2)')
    name_list.append(('Lev', 'int'))

    parser.add_argument('--s', type=float, default=1,    #6
                        help='dilation scale > 1 (default: 2)')
    name_list.append(('s', 'float'))

    parser.add_argument('--n', type=int, default=2,     #7
                        help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
    name_list.append(('n', 'int'))

    parser.add_argument('--FrameType', type=str, default='Linear',
                        help='frame type (default: Entropy): Si1.29gmoid, Entropy')
    name_list.append(('FrameType', 'str'))
    
    parser.add_argument('--method', type=int, default=0,  help='Model 0 or 1 or 2')
    name_list.append(('method', 'int'))
 #%%   
    parser.add_argument('--runs', type=int, default=10, help='Number of repeating experiments.')
    name_list.append(('runs', 'int'))

    parser.add_argument('--epochs', type=int, default=70, help='Number of epochs to train.')
    name_list.append(('epochs', 'int'))

    parser.add_argument('--lr', type=float, default=0.01,  help='Initial learning rate.') #0.005,   #0.01
    name_list.append(('lr', 'float'))

    parser.add_argument('--weight_decay', type=float, default=5e-4,help='Weight decay (L2 loss on parameters).')
    name_list.append(('weight_decay', 'float'))

    parser.add_argument('--num_hid', type=int, default=32, help='Number of hidden units.')#32,  #16,
    name_list.append(('num_hid', 'int'))

    parser.add_argument('--dropout', type=float, default=0.75, help='Dropout rate (1 - keep probability).')
    name_list.append(('dropout', 'float'))

    parser.add_argument('--K', type=int, default=10,  help='K.')  #2
    name_list.append(('K', 'int'))

    parser.add_argument('--warmup', type=int, default=10, help='Warm-up steps: default 10') # 2
    name_list.append(('warmup', 'int'))

    parser.add_argument('--alpha',  type=float,  default=0.5, help='alpha.')
    name_list.append(('alpha', 'float'))

    parser.add_argument('--seed', type=int, default=33129394,
                        help='random seed (default: 1000)')
    name_list.append(('seed', 'int'))
    
    parser.add_argument('--ratio', type=float, default=0.00,
                        help='Add Gaussian nosie Sigma')
    name_list.append(('noise_ratio', 'float'))
    

    #parser.add_argument('--Init',
    #                    type=str,
    #                    default='PPR',
    #                    choices=['SGC', 'PPR', 'NPPR', 'Random', 'WS', 'Null'])
    #parser.add_argument('--Gamma',
    #                    default=None)
    #parser.add_argument('--ppnp',
    #                    type=str,
    #                    default='GPR_prop',
    #                    choices=['PPNP', 'GPR_prop'])
    #parser.add_argument('--dprate',
    #                    type=float,
    #                    default=0.75)
    # parser.add_argument('--num_heads',
    #                    type=int,
    #                    default=1,
    #                    help='Number of heads.')
    args = parser.parse_args()
    #overwrite arguments
    #args.input, args.method, args.mu, args.p, args.s, args.n = data, method, mu, p, s, n

    return args, name_list

if __name__ == '__main__':
    main(*get_args())

# For cora results:
# Namespace(FrameType='Linear', Gamma=None, Init='PPR', K=4, Lev=1, alpha=0.0, dprate=0.75, dropout=0.75, epochs=20, input='cora', lr=0.01, method=2, model='framelet_pgnn', mu=0.5, n=3, num_heads=1, num_hid=32, p=1.5, ppnp='GPR_prop', runs=10, s=6, seed=33129394, train_rate=0.2, val_rate=0.1, weight_decay=0.0005)
# tensor([85.1813, 84.4982, 85.3915, 85.4440, 85.2864, 85.1287, 85.0237, 84.6033, 85.2864, 85.2338])
# Averaged test accuracy for 10 runs: 85.11 \pm 0.32

# Namespace(FrameType='Linear', Gamma=None, Init='PPR', K=4, Lev=1, alpha=0.0, dprate=0.75, dropout=0.75, epochs=20, input='cora', lr=0.01, method=2, model='framelet_pgnn', mu=0.5, n=3, num_heads=1, num_hid=32, p=2.5, ppnp='GPR_prop', runs=10, s=6, seed=33129394, train_rate=0.2, val_rate=0.1, weight_decay=0.0005)
# tensor([84.7084, 85.6542, 84.7609, 85.9170, 85.2338, 85.2864, 84.8135, 84.7084, 85.4966, 85.1287])
# Averaged test accuracy for 10 runs: 85.17 \pm 0.43


# Method 1  (please change the two lines in framelet_conv.py)
# k = 3, p=1.5, epoch = 150, nhid = 32, runs = 4
# tensor([79.3184, 78.6987, 79.5507, 79.1634])
# Averaged test accuracy for 4 runs: 78.72 \pm 0.47
# k =4, p = 2.5, epoch =150, nhid = 16
# tensor([78.0015, 78.9698, 79.2409, 78.6600])
# Averaged test accuracy for 4 runs: 78.72 \pm 0.46

# Method 2  (lr = 0.005, droppout = 0.75)
# k =4, p = 2.5, epoch =150, nhid = 32
# tensor([79.7057, 78.0015, 79.0860, 79.5895, 78.0015])
# Averaged test accuracy for 5 runs: 78.88 \pm 0.83
# k =4, p = 2.5, epoch =80, nhid = 32
# tensor([79.7057, 78.9698, 79.4345, 77.5755, 78.8149])
# Averaged test accuracy for 5 runs: 78.90 \pm 0.82
# k =4, p = 3.0, epoch =150, nhid = 32
# tensor([79.2409, 78.0403, 78.6212, 79.5120, 78.3114])
# Averaged test accuracy for 5 runs: 78.75 \pm 0.62
# k =4, p = 3.0, epoch = 80, nhid = 32
# tensor([79.2409, 78.7374, 78.9311, 77.4593, 79.0085])
# Averaged test accuracy for 5 runs: 78.68 \pm 0.70

# Method 3  (pre-K = 10)
#Namespace(FrameType='Linear', Gamma=None, Init='PPR', K=5, Lev=2, alpha=0.0, dprate=0.75, dropout=0.75, epochs=70, input='cora', lr=0.005, model='framelet_pgnn', mu=0.1, n=2, num_heads=8, num_hid=32, p=2.5, ppnp='GPR_prop', runs=5, s=2, seed=129394, train_rate=0.025, val_rate=0.025, weight_decay=0.0005)
#tensor([79.3184, 79.1634, 78.9311, 78.7761, 78.8923])
#Averaged test accuracy for 5 runs: 79.02 \pm 0.22


#Namespace(FrameType='Haar', Gamma=None, Init='PPR', K=4, Lev=1, alpha=0.0, dprate=0.75, dropout=0.7, epochs=150, input='wisconsin', lr=0.005, method=2, model='framelet_pgnn', mu=70.0, n=7, num_heads=8, num_hid=32, p=2.5, ppnp='GPR_prop', runs=10, s=6, seed=129394, train_rate=0.6, val_rate=0.2, weight_decay=0.0005)
#tensor([90.7407, 90.7407, 91.6667, 95.3704, 94.4444, 94.4444, 97.2222, 93.5185,91.6667, 95.3704])
#Averaged test accuracy for 10 runs: 93.52 \pm 2.23
#Namespace(FrameType='Linear', Gamma=None, Init='PPR', K=5, Lev=1, alpha=0.0, dprate=0.75, dropout=0.75, epochs=70, input='wisconsin', lr=0.005, method=2, model='framelet_pgnn', mu=70.0, n=7, num_heads=1, num_hid=32, p=2.5, ppnp='GPR_prop', runs=10, s=6, seed=33129394, train_rate=0.6, val_rate=0.2, weight_decay=0.0005)
#tensor([93.5185, 95.3704, 96.2963, 92.5926, 96.2963, 95.3704, 97.2222, 96.2963, 93.5185, 97.2222])
#Averaged test accuracy for 10 runs: 95.37 \pm 1.63