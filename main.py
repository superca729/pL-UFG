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
    parser.add_argument('--input', type=str, default='cornell', help='Input graph.')  #['cora', 'citeseer', 'pubmed']['cornell', 'texas', 'wisconsin']
    name_list.append(('input', 'str'))

    parser.add_argument('--train_rate', type=float, default=0.6, help='Training rate.')
    name_list.append(('train_rate', 'float'))

    parser.add_argument('--val_rate', type=float, default=0.2, help='Validation rate.') 
    name_list.append(('val_rate', 'float'))

    parser.add_argument('--model', type=str, default='ufg_pgnn',
                        choices=['ufg','pgnn', 'ufg_pgnn', 'mlp', 'gcn', 'cheb', 'sgc', 'gat', 'jk', 'appnp', 'gprgnn'],
                        help='GNN model')
    name_list.append(('model', 'str'))
#%%    
    parser.add_argument('--mu', type=float, default=50, help='mu.') 
    name_list.append(('mu', 'float'))

    parser.add_argument('--p', type=float, default=1, help='p.') 
    name_list.append(('p', 'float'))

    parser.add_argument('--Lev', type=int, default=1, help='level of transform (default: 2)')
    name_list.append(('Lev', 'int'))

    parser.add_argument('--s', type=float, default=1,    
                        help='dilation scale > 1 (default: 2)')
    name_list.append(('s', 'float'))

    parser.add_argument('--n', type=int, default=6,     
                        help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
    name_list.append(('n', 'int'))

    parser.add_argument('--FrameType', type=str, default='Linear',
                        help='frame type (default: Linear): Haar')
    name_list.append(('FrameType', 'str'))
    
    parser.add_argument('--method', type=int, default=0,  help='Model 0 or 1 or 2')
    name_list.append(('method', 'int'))
 #%%   
    parser.add_argument('--runs', type=int, default=10, help='Number of repeating experiments.')
    name_list.append(('runs', 'int'))

    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs to train.')
    name_list.append(('epochs', 'int'))

    parser.add_argument('--lr', type=float, default=0.01,  help='Initial learning rate.') #0.005,   #0.01
    name_list.append(('lr', 'float'))

    parser.add_argument('--weight_decay', type=float, default=5e-4,help='Weight decay (L2 loss on parameters).')
    name_list.append(('weight_decay', 'float'))

    parser.add_argument('--num_hid', type=int, default=32, help='Number of hidden units.')#32,  #16,
    name_list.append(('num_hid', 'int'))

    parser.add_argument('--dropout', type=float, default=0.75, help='Dropout rate (1 - keep probability).')
    name_list.append(('dropout', 'float'))

    parser.add_argument('--K', type=int, default=4,  help='K.')  
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

    return args, name_list

if __name__ == '__main__':
    main(*get_args())



