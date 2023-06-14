from os.path import join
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tqdm import tqdm
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import global_add_pool
import schnetpack as spk

from model.model_3d.painn import PaiNN, EquivariantScalar
from model.model_3d.schnet import SchNet
from model.mlp import MLP
from datasets.qm9_3d import QM93D

meann, mad = 0, 1
def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def compute_mean_mad(values):
    meann = torch.mean(values)
    mad = torch.std(values)
    return meann, mad

def train_general(model, loader, optimizer):
    model.train()
    output_layer.train()
    total_loss = 0   
    for step, batch in enumerate(tqdm(loader)):
        batch = batch.to(device)

        if args.model == 'schnet': 
            molecule_repr, node_repr = model(batch.z, batch.pos, batch.batch, t_emb=None) 
            pred = output_layer(molecule_repr)

        elif args.model == 'painn':
            scalar, vector = model(batch.z, batch.pos, batch.batch)
            pred = global_add_pool(output_layer.pre_reduce(scalar, vector), batch.batch)
        
        
        y = batch.y.view(pred.shape).float()
        y = ((y-meann)/mad)
        loss = reg_criterion(pred, y)

        # ===== warmup for CosineAnnealingLR =====
        global warmup_steps
        warmup_steps += 1
        if warmup_steps < args.lr_warmup_steps:
            lr_scale = min(
                1.0,
                float(warmup_steps)
                / float(args.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * args.lr
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def eval_general(model, loader):
    model.eval()
    y_true, y_pred = [], []

    for step, batch in enumerate(loader):
        with torch.no_grad():
            batch = batch.to(device)

            if args.model == 'schnet':
                output_layer.eval()
                molecule_repr, _ = model(batch.z, batch.pos, batch.batch)  
                pred = output_layer(molecule_repr)

            elif args.model == 'painn':
                output_layer.eval()
                scalar, vector = model(batch.z, batch.pos, batch.batch)
                pred = global_add_pool(output_layer.pre_reduce(scalar, vector), batch.batch)
    
        true = batch.y.view(pred.shape).float()
        y_true.append(true)
        y_pred.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_pred = (torch.cat(y_pred, dim=0)*mad + meann).cpu().numpy()

    
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae}, y_true, y_pred


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # ========= Seed and basic info ==========
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--runseed', type=int, default=1)
    parser.add_argument('--device', type=int, default=7)

    # ========= Hyper-parameters ===========
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--decay', type=float, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--finetune_ratio', type=float, default=0.1)

    # ========= Hyper-parameters ===========
    parser.add_argument('--lr_decay_step_size', type=int, default=150)
    parser.add_argument('--lr_decay_factor', type=int, default=0.5)
    parser.add_argument('--lr_cosine_length', type=int, default=400000, help='Cosine length if lr_schedule is cosine.')
    parser.add_argument('--lr_warmup_steps', type=int, default=1e4, help='Warm-up Steps.')
    parser.add_argument('--patience', type=int, default=30, help='Early stopping patiance.')
    parser.add_argument('--decay_patience', type=int, default=5, help='Scheduler decay patiance.')
    parser.add_argument('--decay_factor', type=float, default=0.5, help='Scheduler decay patiance.')

    # ======== Model configuration =========
    parser.add_argument('--model', type=str, default='painn')
    parser.add_argument('--property', type=str, default='lumo')

    # ======== SchNet hyperparameters ==========
    parser.add_argument('--num_filters', type=int, default=128)
    parser.add_argument('--num_interactions', type=int, default=6)
    parser.add_argument('--num_gaussians', type=int, default=51)
    parser.add_argument('--cutoff', type=float, default=5.)
    parser.add_argument('--readout', type=str, default='add',
                        choices=['mean', 'add'])
    
    # ========= Program viewing =========== 
    parser.add_argument('--eval_train', dest='eval_train', action='store_true')
    parser.add_argument('--no_eval_train', dest='eval_train', action='store_false')
    parser.set_defaults(eval_train=False)
    
    args = parser.parse_args()
    seed_all(args.runseed)
    device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    num_tasks = 1
    target = args.property
    dataset = QM93D()
    dataset.data.y = dataset.data[target]
    print('The total number of molecules of QM9: {}'.format(dataset.data.y.shape[0])) 
    
    if target in ['U','U0']: args.lr = 1e-5

    split_idx = dataset.get_idx_split(len(dataset.data.y), train_size=110000, valid_size=10000, seed=42)
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']], dataset[split_idx['test']]
    
    # ====== Data Selection =======
    num_mols = len(train_dataset)
    random.seed(args.runseed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)
    ids = all_idx[:int(args.finetune_ratio * num_mols)]
    train_dataset = train_dataset[ids]
    print(train_dataset)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    meann, mad = compute_mean_mad(train_dataset.data.y)
    model_param_group = []
    # Model Configuration
    if args.model == 'schnet':
        model = SchNet(hidden_channels=args.hidden_dim, num_filters=args.num_filters, num_interactions=args.num_interactions,
                                        num_gaussians=args.num_gaussians, cutoff=args.cutoff, atomref=None, readout=args.readout,
                                        dipole=False).to(device)
        output_layer = MLP(args.hidden_dim, args.hidden_dim, num_tasks, 1).to(device)
        model_param_group.append({'params': output_layer.parameters(), 'lr': args.lr})

    elif args.model == 'painn':
        radial_basis = spk.nn.GaussianRBF(n_rbf=20, cutoff=args.cutoff)
        model = PaiNN(n_atom_basis=args.hidden_dim, n_interactions=args.num_interactions, radial_basis=radial_basis,
                        cutoff_fn=spk.nn.CosineCutoff(args.cutoff)).to(device)
        output_layer = EquivariantScalar(hidden_channels=args.hidden_dim).to(device)
        model_param_group.append({'params': output_layer.parameters(), 'lr': args.lr})
    print('======= Model Loaded =======')
    print(model)

    
    model_param_group.append({'params': model.parameters(), 'lr': args.lr})

    optimizer = optim.Adam(model_param_group, lr=args.lr, weight_decay=args.decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.lr_cosine_length)
    warmup_steps = 0
    reg_criterion = torch.nn.MSELoss()

    train_result_list, val_result_list, test_result_list = [], [], []
    metric_list = ['RMSE', 'MAE']
    best_val_mae, best_val_idx = 1e10, 0

    train_func = train_general
    eval_func = eval_general
    es = 0

    for epoch in range(1, args.epochs + 1):
        loss_acc = train_func(model, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        if args.eval_train:
            train_result, train_target, train_pred = eval_func(model, train_loader)
        else:
            train_result = {'RMSE': 0, 'MAE': 0, 'R2': 0}
        val_result, val_target, val_pred = eval_func(model, valid_loader)
        test_result, test_target, test_pred = eval_func(model, test_loader)
        test_result_list.append(test_result)

        train_result_list.append(train_result)
        val_result_list.append(val_result)

        for metric in metric_list:
            print('{} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(metric, train_result[metric], val_result[metric], test_result[metric]))

        if val_result['MAE'] < best_val_mae:
            best_val_mae = val_result['MAE']
            best_val_idx = epoch - 1
            

    for metric in metric_list:
        print('Best (RMSE), {} train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(
            metric, train_result_list[best_val_idx][metric], val_result_list[best_val_idx][metric], test_result_list[best_val_idx][metric]))