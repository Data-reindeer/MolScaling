from pathlib import Path
import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.contrib import tzip
import os
from functools import partial

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from config import args
from datasets.pcqm4mv2 import PygPCQM4Mv2Dataset
from datasets.dataloader import DataLoaderMaskingPred
from model.gnn import GNN, GNNDecoder

def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]\n")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def save_model(save_best):
    if not args.output_model_dir == '':
        if save_best:
            global optimal_loss
            print('save model with loss: {:.5f}'.format(optimal_loss))
            path = args.output_model_dir 
            if not os.path.exists(path):  os.makedirs(path)
            torch.save(model.state_dict(), path + 'PubChem_Pretrained.pth')

    return

def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def recons_loss(batch, model, dec_atom):
    criterion = partial(sce_loss, alpha=1)
    node_repr = model(batch)
    mask_idx = batch.masked_atom_indices
    
    node_label = batch.node_attr_label
    pred_node = dec_atom(node_repr, batch.edge_index, batch.edge_attr, mask_idx)
    loss = 0
    loss += criterion(node_label, pred_node[mask_idx])

    return loss

def train(args, model_list, device, loader, optimizer_list):   
    start_time = time.time()
    model, dec_pred_atoms = model_list
    optimizer_model, optimizer_dec_pred_atoms = optimizer_list
    model.train()
    dec_pred_atoms.train()

    loss_accum = 0
    l = tqdm(loader, total=len(loader))
        
    for step, batch_mol in enumerate(l):    
        batch_mol = batch_mol.to(device)

        # ===== Calculate reconstruction Loss =====
        loss = recons_loss(batch_mol, model, dec_pred_atoms)

        optimizer_model.zero_grad()
        optimizer_dec_pred_atoms.zero_grad()

        loss.backward()

        optimizer_model.step()
        optimizer_dec_pred_atoms.step()

        loss_accum += float(loss.cpu().item())

    global optimal_loss
    loss_accum /= len(loader)

    temp_loss = loss_accum
    if temp_loss < optimal_loss:
        optimal_loss = temp_loss
        save_model(save_best=True)

    print('Total Loss: {:.5f}\tTime: {:.5f}'.format(
        loss_accum, time.time() - start_time))

    return 

if __name__ == '__main__':
    print('===start===\n')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seed_all(args.seed)

    dataset = PygPCQM4Mv2Dataset()
    split_idx = dataset.get_idx_split()
    train_dataset, valid_dataset, test_dataset = dataset[split_idx['train']], dataset[split_idx['valid']],dataset[split_idx['test-dev']]
    print('===== Dataset Loaded =====')
                              
    loader = DataLoaderMaskingPred(train_dataset, batch_size=args.batch_size, shuffle=True, mask_rate=args.mask_rate,
                                    num_workers = args.num_workers)
    NUM_NODE_ATTR = 119
    
    model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim).to(device)
    atom_pred_decoder = GNNDecoder(args.emb_dim, NUM_NODE_ATTR, JK=args.JK, gnn_type=args.net2d).to(device)

    
    model_list = [model, atom_pred_decoder] 
    
    optimizer_model = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_dec_pred_atoms = optim.Adam(atom_pred_decoder.parameters(), lr=args.lr, weight_decay=args.decay)
    optimizer_list = [optimizer_model, optimizer_dec_pred_atoms]
    optimal_loss = 1e10
    

    for epoch in range(1, args.epochs + 1):
        print('epoch: {}'.format(epoch))
        train(args, model_list, device, loader, optimizer_list)
    
