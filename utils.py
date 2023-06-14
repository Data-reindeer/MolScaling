from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.utils.metric import type_metric, distance_metric
import numpy as np
from rdkit import DataStructs
import random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

import pdb
# ======= Kmeans Clustering for molecular fingerprints ========
# def fp_similarity(fp1, fp2):
#     if type(fp1) is not np.ndarray: pdb.set_trace()
#     return DataStructs.TanimotoSimilarity(fp1, fp2)
def reorder(cluster_labels):
    length = sum(len(l) for l in cluster_labels)
    labels = [0 for _ in range(length)]
    for c in range(len(cluster_labels)):
        for idx in cluster_labels[c]:
            labels[idx] = c
    return labels

def fp_similarity(p_vec, q_vec):
    pq = np.dot(p_vec, q_vec)
    p_square = np.linalg.norm(p_vec)
    q_square = np.linalg.norm(q_vec)
    return pq / (p_square + q_square - pq + 1e-5)

def Kmeans(fps, K, device):
    # Distance Definition 
    metric = distance_metric(type_metric.USER_DEFINED, func=fp_similarity)

    # Compute start centers
    # start_centers = cal_centers(fps, K, seed)
    start_centers = kmeans_plusplus_initializer(fps, K).initialize()
    pdb.set_trace()

    # create K-Means algorithm
    kmeans_instance = kmeans(fps, start_centers, metric=metric)
    kmeans_instance.process()
    cluster_labels = reorder(kmeans_instance.get_clusters())
    centers = kmeans_instance.get_centers()
    scores = [fp_similarity(centers[cluster_labels[idx]], fps[idx]) for idx in range(len(fps))]

    return scores, cluster_labels

# ========== Data Pruning ============
def PrototypesGetHardExamples(cosine_scores, cluster_labels, uids, return_size, balance=0.5):
    n = len(cosine_scores)
    assert(len(cosine_scores) == len(cluster_labels))
    assert(len(cosine_scores) == len(uids))
    
    k = max(cluster_labels) + 1 ## +1 for zero-indexed
    assert(return_size >= 1 and return_size <= n)

    print("Creating clusters:")
    clusters = [[] for i in range(k)]
    for i in tqdm(range(n)):
        clusters[cluster_labels[i]].append( (cosine_scores[i], uids[i]) )

    print("Getting minimum numbers for each cluster:")
    returning = []
    leftovers = []
    for i in tqdm(range(k)):
        cluster = clusters[i]
        cluster.sort()

        soft_min_num = balance * len(cluster) * return_size / n
        min_num = int(soft_min_num + 0.99999999)

        returning.extend( cluster[:min_num] )
        leftovers.extend( cluster[min_num:] )    
    
    return_uids = [element[1] for element in returning]
    
    
    if return_size > len(return_uids):
        print("start sorting leftovers")
        leftover_scores = np.array([element[0] for element in leftovers])
        leftover_uids = np.array([element[1] for element in leftovers])
        top_leftovers = np.argsort(leftover_scores)[:(return_size - len(return_uids))]
        print("finish sorting leftovers")
        return_uids.extend(list(leftover_uids[top_leftovers]))
    elif len(return_uids) > return_size:
        print("WARNING: not meeting cluster balancing minimums")
        return_uids = random.sample(return_uids, return_size)


    return return_uids

# ====== Contrast Pretraining =====
def do_CL(X, Y, args):
    if args.normalize:
        X = F.normalize(X, dim=-1)
        Y = F.normalize(Y, dim=-1)

    criterion = nn.CrossEntropyLoss()
    B = X.size()[0]
    logits = torch.mm(X, Y.transpose(1, 0))  # B*B
    logits = torch.div(logits, args.T)
    labels = torch.arange(B).long().to(logits.device)  # B*1

    CL_loss = criterion(logits, labels)
    pred = logits.argmax(dim=1, keepdim=False)
    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / B

    return CL_loss, CL_acc
def dual_CL(X, Y, args):
    CL_loss_1, CL_acc_1 = do_CL(X, Y, args)
    CL_loss_2, CL_acc_2 = do_CL(Y, X, args)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2

def do_CL_info(c, X, Y, args):
    X = F.normalize(X, dim=-1)
    Y = F.normalize(Y, dim=-1)

    criterion = nn.CrossEntropyLoss()
    B = int(X.size()[0] / c)
    batch = torch.arange(B)
    idx = batch.repeat_interleave(c).long().to(X.device)

    logits = torch.mm(X, Y.transpose(1, 0))  # Bc*Bc
    logits = torch.div(logits, args.T)

    multi_logits = torch.zeros(B, B).to(X.device)
    tmp = torch.zeros(B*c, B).to(X.device)
    tmp = scatter_add(logits, idx, out=tmp, dim=1)
    multi_logits = scatter_add(tmp, idx, out=multi_logits, dim=0)

    labels = torch.arange(B).long().to(multi_logits.device)  # B*1

    CL_loss = criterion(multi_logits, labels)
    pred = multi_logits.argmax(dim=1, keepdim=False)
    CL_acc = pred.eq(labels).sum().detach().cpu().item() * 1. / (B)
    
    return CL_loss, CL_acc

def dual_CL_info(c, X, Y, args):
    CL_loss_1, CL_acc_1 = do_CL_info(c, X, Y, args)
    CL_loss_2, CL_acc_2 = do_CL_info(c, Y, X, args)
    return (CL_loss_1 + CL_loss_2) / 2, (CL_acc_1 + CL_acc_2) / 2

# ====== Dataset for Fingerprint Kmeans ======
import torch
from torch.utils.data import Dataset, DataLoader

class FingerprintDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return torch.tensor(self.data[index])

    def __len__(self):
        return len(self.data)

import matplotlib.pyplot as plt   
def plot_distribution(dataset):
    cnts = []
    for mol in dataset:
        cnts.append(mol.num_nodes)
    plt.hist(cnts, bins=50, range=(0,50))
    plt.savefig("./pcba_dis.png")
    return cnts