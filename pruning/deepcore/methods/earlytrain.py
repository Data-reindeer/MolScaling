from .coresetmethod import CoresetMethod
import torch, time
from torch import nn
import numpy as np
from copy import deepcopy
from torch_geometric.loader import DataLoader
from model.mlp import MLP
from model.gnn import GNN
from torch_geometric.nn import global_mean_pool
import pdb


class EarlyTrain(CoresetMethod):
    '''
    Core code for training related to coreset selection methods when pre-training is required.
    '''

    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, num_tasks=None, epochs=200,
                  dst_test=None, device=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, num_tasks, device)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.model = GNN(num_layer=args.num_layer, emb_dim=args.emb_dim, drop_ratio=args.dropout_ratio).to(self.device)
        self.output_layer = MLP(in_channels=args.emb_dim, hidden_channels=args.emb_dim, 
                        out_channels=self.num_classes, num_layers=1, dropout=0).to(self.device)

        self.dst_test = dst_test

    def train(self, epoch, **kwargs):
        """ Train model for one epoch """

        self.before_train()
        self.model.train()
        self.output_layer.train()

        print('\n=> Training Epoch #%d' % epoch)
        list_of_train_idx = list(range(self.n_train))
        trainset_permutation_inds = np.random.permutation(list_of_train_idx)
        batch_sampler = torch.utils.data.BatchSampler(trainset_permutation_inds, batch_size=self.args.batch_size,
                                                      drop_last=False)
        trainset_permutation_inds = list(batch_sampler)
        train_loader = DataLoader(self.dst_train, batch_size=self.args.batch_size,
                              shuffle=True, num_workers=0)

        for i, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            h = global_mean_pool(self.model(batch), batch.batch)
            outputs = self.output_layer(h)
            targets = batch.y.view(outputs.shape).to(torch.float64)

            # Forward propagation, compute loss, get predictions
            self.model_optimizer.zero_grad()
            # Whether y is non-null or not.
            is_valid = targets ** 2 > 0
            # Loss matrix
            loss_mat = self.criterion(outputs.double(), (targets + 1) / 2)
            # loss matrix after removing null target
            loss_mat = torch.where(
                is_valid, loss_mat,
                torch.zeros(loss_mat.shape).to(self.device).to(loss_mat.dtype))
            # loss = self.criterion(outputs, targets)
            loss = torch.sum(loss_mat) / torch.sum(is_valid)
            self.after_loss(outputs, loss, targets, trainset_permutation_inds[i], epoch)

            # # Update loss, backward propagate, update optimizer
            loss = loss.mean()

            self.while_update(outputs, loss, targets, epoch, i, self.args.batch_size)

            loss.backward()
            self.model_optimizer.step()
        return self.finish_train()

    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.train_indx = np.arange(self.n_train)

        if self.device == "cpu":
            print("===== Using CPU =====")
        else:
            print("===== Using GPU =====")

        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

        # Setup optimizer
        model_param_group = []
        model_param_group.append({'params': self.output_layer.parameters(),'lr': self.args.lr})
        model_param_group.append({'params': self.model.parameters(), 'lr': self.args.lr})
        self.model_optimizer = torch.optim.Adam(model_param_group, lr=self.args.selection_lr,
                                                    weight_decay=self.args.selection_decay)

        self.before_run()

        for epoch in range(self.epochs):
            self.before_epoch()
            self.train(epoch)
            if self.dst_test is not None and self.args.selection_test_interval > 0 and (
                    epoch + 1) % self.args.selection_test_interval == 0:
                self.test(epoch)
            self.after_epoch()

        return self.finish_run()

    def test(self, epoch):
        self.model.no_grad = True
        self.model.eval()
        self.output_layer.eval()

        test_loader = DataLoader(self.dst_test, batch_size=self.args.batch_size,
                              shuffle=False, num_workers=self.args.num_workers)
        correct = 0.
        total = 0.

        print('\n=> Testing Epoch #%d' % epoch)

        for batch_idx, (input, target) in enumerate(test_loader):
            output = self.model(input.to(self.device))
            loss = self.criterion(output, target.to(self.device)).sum()

            predicted = torch.max(output.data, 1).indices.cpu()
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % self.print_freq == 0:
                print('| Test Epoch [%3d/%3d] Iter[%3d/%3d]\t\tTest Loss: %.4f Test Acc: %.3f%%' % (
                    epoch, self.epochs, batch_idx + 1, (round(len(self.dst_test) * self.args.selection_test_fraction) //
                                                        self.args.batch_size) + 1, loss.item(),
                    100. * correct / total))

        self.model.no_grad = False

    def num_classes_mismatch(self):
        pass

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_idx, epoch):
        pass

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        pass

    def finish_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def finish_run(self):
        pass

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
