from os.path import join
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from tqdm import tqdm
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from splitter import scaffold_split, random_split, imbalanced_split
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig

from config import args
from datasets.molnet import MoleculeDataset
from model.fp_model import SmilesEncoder
from model.mlp import MLP

def seed_all(seed):
    if not seed:
        seed = 0
    print("[ Using Seed : ", seed, " ]")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

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

def train_general(model, device, loader, optimizer):
    model.train()
    output_layer.train()
    total_loss = 0

    for step, batch in enumerate(tqdm(loader)):
        ids = batch[0].to(device)
        att_mask = batch[1].to(device)
        y = batch[2].to(device)
        
        if args.net_sm == 'transformer':
            molecule_repr = model(ids, att_mask)
        elif args.net_sm == 'RoBERTa':
            molecule_repr = model(input_ids=ids, attention_mask=att_mask).pooler_output
        pred = output_layer(molecule_repr)

        y = y.view(pred.shape).to(torch.float64)
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
        global warmup_steps
        warmup_steps += 1
        if warmup_steps < args.lr_warmup_steps * args.finetune_ratio:
            lr_scale = min(
                1.0,
                float(warmup_steps)
                / float(args.lr_warmup_steps),
            )

            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * roberta_lr
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()

    return total_loss / len(loader)


def eval_general(model, device, loader):
    model.eval()
    output_layer.eval()
    y_true, y_scores = [], []
    total_loss = 0

    for step, batch in enumerate(loader):
        ids = batch[0].to(device)
        att_mask = batch[1].to(device)
        y = batch[2].to(device)
        with torch.no_grad():
            if args.net_sm == 'transformer':
                molecule_repr = model(ids, att_mask)
            elif args.net_sm == 'RoBERTa':
                molecule_repr = model(input_ids=ids, attention_mask=att_mask).pooler_output

            pred = output_layer(molecule_repr)
    
        true = y.view(pred.shape)

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
                ap = average_precision_score(y_true[is_valid,i], y_scores[is_valid,i])

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
    dataset_folder = '/home/chendingshuo/MoD/datasets/molecule_net/'
    dataset = MoleculeDataset(dataset_folder + args.dataset, dataset=args.dataset)
    print(dataset)

    eval_metric = roc_auc_score

    if args.split == 'scaffold':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles), idx_tuple = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, return_smiles=True)
        print('split via scaffold')
    elif args.split == 'random':
        smiles_list = pd.read_csv(dataset_folder + args.dataset + '/processed/smiles.csv',
                                  header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles), idx_tuple = random_split(
            dataset, null_value=0, frac_train=0.8, frac_valid=0.1,
            frac_test=0.1, seed=args.seed, smiles_list=smiles_list)
        print('randomly split')
    else:
        raise ValueError('Invalid split option.')
    
    num_mols = len(train_dataset)
    print('Trainig samples: {}'.format(num_mols))
    random.seed(args.runseed)
    all_idx = list(range(num_mols))
    random.shuffle(all_idx)
    ids = all_idx[:int(args.finetune_ratio * num_mols)]
    train_dataset = train_dataset[ids]
    train_smiles = [train_smiles[i] for i in ids]

    # ======== Tokenize SMILES strings =========
    tokenizer = RobertaTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    train_dict = tokenizer(train_smiles,return_tensors="pt", padding='max_length', truncation=True, max_length=510)
    valid_dict = tokenizer(valid_smiles,return_tensors="pt", padding='max_length', truncation=True, max_length=510)
    test_dict = tokenizer(test_smiles,return_tensors="pt", padding='max_length', truncation=True, max_length=510)
    full_y = dataset.data.y.view(-1,num_tasks)

    train_dataset = TensorDataset(train_dict['input_ids'], train_dict['attention_mask'], full_y[torch.tensor([idx_tuple[0][i] for i in ids])])
    valid_dataset = TensorDataset(valid_dict['input_ids'], valid_dict['attention_mask'], full_y[torch.tensor(idx_tuple[1])])
    test_dataset = TensorDataset(test_dict['input_ids'], test_dict['attention_mask'], full_y[torch.tensor(idx_tuple[2])])

    train_loader = DataLoader(train_dataset, batch_size=32,
                              shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=32,
                            shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=32,
                             shuffle=False, num_workers=args.num_workers)
    
    if args.net_sm == 'transformer':
        model = SmilesEncoder(vocab_size=tokenizer.vocab_size, word_dim=64, out_dim=args.emb_dim, num_layer=1).to(device)
        output_layer = MLP(args.emb_dim, args.emb_dim, num_tasks, 1).to(device)
        roberta_lr = 1e-3 


    elif args.net_sm == 'RoBERTa':
        config = RobertaConfig(
            vocab_size = tokenizer.vocab_size,
            hidden_act = "gelu",
            hidden_dropout_prob = 0.1,
            hidden_size = 768,
            initializer_range = 0.02,
            intermediate_size = 3072,
            layer_norm_eps = 1e-12,
            model_type = "roberta",
            num_attention_heads = 12,
            num_hidden_layers = 6,
            type_vocab_size = 1,
            )
        model = RobertaModel(config).to(device)
        output_layer = MLP(config.hidden_size, args.emb_dim, num_tasks, 1).to(device)
        roberta_lr = 7e-5
    
    else:
        raise ValueError('Invalid model option.')  
    print(model)
     
    warmup_steps = 0
    model_param_group = []
    model_param_group.append({'params': model.parameters(), 'lr': roberta_lr})
    model_param_group.append({'params': output_layer.parameters(), 'lr': roberta_lr})
                  
    optimizer = optim.AdamW(model_param_group, lr=roberta_lr, betas=(0.9, 0.98), eps=1e-6, 
                           weight_decay=0.01)

    criterion = nn.BCEWithLogitsLoss(reduction='none')
    train_roc_list, val_roc_list, test_roc_list = [], [], []
    roc_lists = []
    best_val_roc, best_val_idx = -1, 0

    train_func = train_general
    eval_func = eval_general

    for epoch in range(1, args.epochs + 1):
        loss_acc = train_func(model, device, train_loader, optimizer)
        print('Epoch: {}\nLoss: {}'.format(epoch, loss_acc))

        train_roc = train_acc = 0
        
        val_roc, val_acc, _ = eval_func(model, device, val_loader)
        test_roc, test_acc, roc_list = eval_func(model, device, test_loader)

        train_roc_list.append(train_roc)
        val_roc_list.append(val_roc)
        test_roc_list.append(test_roc)
        roc_lists.append(roc_list)

        print('train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc, val_roc, test_roc))
        print()

        if val_roc > best_val_roc:
            best_val_roc = val_roc
            best_val_idx = epoch - 1

    print('best train: {:.6f}\tval: {:.6f}\ttest: {:.6f}'.format(train_roc_list[best_val_idx], val_roc_list[best_val_idx], test_roc_list[best_val_idx]))
    print('single tasks roc list:{}'.format(roc_lists[best_val_idx]))
