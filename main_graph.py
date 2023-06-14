from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from rdkit.Chem import AllChem, RDKFingerprint

from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch_geometric.utils import degree 
from splitter import scaffold_split, random_split, imbalanced_split


from config import args
from datasets.molnet import MoleculeDataset
from model.gnn import GNN
from model.mlp import MLP
from utils import PrototypesGetHardExamples
from pruning.mini_kmeans import KMeans
import pruning.deepcore.methods as methods

def compute_degree(train_dataset):
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in train_dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    
    return deg

def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def save_model(save_best):
    dir_name = 'fingerprint/'
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))

            torch.save(model.state_dict(), args.output_model_dir \
                        + dir_name + 'transformer_{}_{}.pth'.format(args.dataset, args.finetune_ratio))

    return

def get_num_task(dataset):
    # Get output dimensions of different tasks
    if dataset == 'tox21':
        return 12
    elif dataset in ['hiv', 'bace', 'bbbp']:
        return 1
    elif dataset == 'muv':
        return 17
    elif dataset == 'toxcast':
        return 617
    elif dataset == 'sider':
        return 27
    elif dataset == 'clintox':
        return 2
    elif dataset == 'pcba':
        return 92
    raise ValueError('Invalid dataset name.')

# TODO: clean up
def train_general(model, device, loader, optimizer):
    model.train()
    total_loss = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        h = global_mean_pool(model(batch), batch.batch)
        pred = output_layer(h)
        
        y = batch.y.view(pred.shape).to(torch.float64)
        # Whether y is non-null or not.
        is_valid = y ** 2 > 0
        # Loss matrix
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        # loss matrix after removing null target
        loss_mat = torch.where(
            is_valid, loss_mat,
            torch.zeros(loss_mat.shape).to(device).to(loss_mat.dtype))
        
        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    global optimal_loss 
    temp_loss = total_loss / len(loader)
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        # save_model(save_best=True)

    return total_loss / len(loader)


def eval_general(model, device, loader):
    model.eval()
    y_true, y_scores = [], []
    total_loss = 0

    for step, batch in enumerate(loader):
        batch = batch.to(device)
        with torch.no_grad():
            h = global_mean_pool(model(batch), batch.batch)
            pred = output_layer(h)
    
        true = batch.y.view(pred.shape)

        y_true.append(true)
        y_scores.append(pred)


    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    if args.dataset == 'pcba':
        ap_list = []

        for i in range(y_true.shape[1]):
            if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
                # ignore nan values
                is_valid = is_valid = y_true[:, i] ** 2 > 0
                ap = average_precision_score(y_true[is_valid, i], y_scores[is_valid, i])

                ap_list.append(ap)

        if len(ap_list) == 0:
            raise RuntimeError('No positively labeled data available. Cannot compute Average Precision.')
        return sum(ap_list) / len(ap_list), total_loss / len(loader), ap_list

    else:
        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
                is_valid = y_true[:, i] ** 2 > 0
                roc_list.append(eval_metric((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))
            else:
                print('{} is invalid'.format(i))

        if len(roc_list) < y_true.shape[1]:
            print(len(roc_list))
            print('Some target is missing!')
            print('Missing ratio: %f' %(1 - float(len(roc_list)) / y_true.shape[1]))

        return sum(roc_list) / len(roc_list), total_loss / len(loader), roc_list


if __name__ == '__main__':
    seed_all(args.runseed)
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    # Bunch of classification tasks
    num_tasks = get_num_task(args.dataset)
    dataset_folder = './datasets/molecule_net/'
    dataset = MoleculeDataset(dataset_folder + args.dataset, dataset=args.dataset)
    
    print(dataset)
    print('=============== Statistics ==============')
    print('Avg degree:{}'.format(torch.sum(degree(dataset.data.edge_index[0])).item()/dataset.data.x.shape[0]))
    print('Avg atoms:{}'.format(dataset.data.x.shape[0]/(dataset.data.y.shape[0]/num_tasks)))
    print('Avg bond:{}'.format((dataset.data.edge_index.shape[1]/2)/(dataset.data.y.shape[0]/num_tasks)))

    eval_metric = roc_auc_score

    if args.split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles), (_,_,_) = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, return_smiles=True)
        print('split via scaffold')
    elif args.split == 'random':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles),_ = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.seed, smiles_list=smiles_list)
        print('randomly split')
    elif args.split == 'imbalanced':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles), (_,_,_) = imbalanced_split(
            dataset, null_value=0, frac_train=0.7, frac_valid=0.15,
            frac_test=0.15, seed=args.seed, smiles_list=smiles_list)
        print('imbalanced split')
        
    else:
        raise ValueError('Invalid split option.')
    print(train_dataset[0])
    print('Training data length: {}'.format(len(train_smiles)))
    finetune_num = int(args.finetune_ratio * len(train_smiles))

    if args.finetune_pruning is True and args.selection == 'Kmeans':
        print('# of SMILES: {}'.format(len(train_smiles)))
        print('===== Converting smiles to mols =====')
        mols = [AllChem.MolFromSmiles(s) for s in tqdm(train_smiles)]
        print('===== Processing fingerprint =====')
        fps = [torch.tensor(RDKFingerprint(mol), dtype=torch.float).unsqueeze(0) for mol in tqdm(mols)]
        fps = torch.cat(fps, dim=0)
        print('====== Fingerprint Finish ! ! ! =======')

        if len(train_smiles) > 1e5: K = 1000
        else: K = 100
        kmeans = KMeans(n_clusters=K, device=device)
        centers = kmeans.fit_predict(fps)
        scores, cluster_labels = kmeans.predict(fps)
        scores = scores.cpu().detach()
        cluster_labels = cluster_labels.cpu().detach()
        ids = PrototypesGetHardExamples(scores, cluster_labels, range(len(fps)), return_size=finetune_num)
        train_dataset = train_dataset[ids]

    elif args.finetune_pruning is True:
        selection_args = dict(epochs=args.selection_epochs,
                                  selection_method=args.uncertainty,
                                  num_tasks=num_tasks)
        method = methods.__dict__[args.selection](dst_train=train_dataset, args=args, fraction=args.finetune_ratio, 
                random_seed=args.runseed, device = device, **selection_args)
        subset = method.select()
        print(len(subset["indices"]))
        train_dataset = train_dataset[subset["indices"]]
    
    else:
        num_mols = len(train_dataset)
        random.seed(args.runseed)
        all_idx = list(range(num_mols))
        random.shuffle(all_idx)
        ids = all_idx[:int(args.finetune_ratio * num_mols)]
        train_dataset = train_dataset[ids]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)            

    # set up model 
    model_param_group = []
    model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.dropout_ratio).to(device)
    output_layer = MLP(in_channels=args.emb_dim, hidden_channels=args.emb_dim, 
                        out_channels=num_tasks, num_layers=1, dropout=0).to(device)
    
    if args.pretrain:
        model_root = 'PubChem_Pretrained.pth'
        model.load_state_dict(torch.load(args.output_model_dir + model_root, map_location='cuda:0'))
        print('======= Model Loaded =======')
    model_param_group.append({'params': output_layer.parameters(),'lr': args.lr})
    model_param_group.append({'params': model.parameters(), 'lr': args.lr})
    
    print(model)                
    optimizer = optim.Adam(model_param_group, lr=args.lr,
                           weight_decay=args.decay)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    train_roc_list, val_roc_list, test_roc_list = [], [], []
    train_loss_list, val_loss_list, test_loss_list = [], [], []
    roc_lists = []
    best_val_roc, best_val_idx = -1, 0
    optimal_loss = 1e10
    es = 0

    train_func = train_general
    eval_func = eval_general

    for epoch in range(1, args.epochs + 1):
        loss_acc = train_func(model, device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        train_roc = train_loss = 0
        
        val_roc, val_loss, _ = eval_func(model, device, val_loader)
        test_roc, test_loss, roc_list = eval_func(model, device, test_loader)

        train_roc_list.append(train_roc)
        val_roc_list.append(val_roc)
        test_roc_list.append(test_roc)
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)
        roc_lists.append(roc_list)
        print('train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc, val_roc, test_roc))
        print()

        if val_roc > best_val_roc:
            best_val_roc = val_roc
            best_val_idx = epoch - 1

    print('best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc_list[best_val_idx], val_roc_list[best_val_idx], test_roc_list[best_val_idx]))
    print('loss train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_loss_list[best_val_idx], val_loss_list[best_val_idx], test_loss_list[best_val_idx]))
    print('single tasks roc list:{}'.format(roc_lists[best_val_idx]))
