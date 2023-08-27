from .earlytrain import EarlyTrain
import torch, time
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import numpy as np
import pdb


class GraNd(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, num_tasks=None, epochs=200, repeat=1,
                balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, num_tasks, epochs, **kwargs)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.repeat = repeat

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))


    def finish_run(self):
        self.model.eval()
        self.output_layer.eval()

        embedding_dim = self.args.emb_dim
        train_loader = DataLoader(self.dst_train, batch_size=self.args.batch_size,
                              shuffle=True, num_workers=0)
        sample_num = self.n_train

        for i, batch in enumerate(train_loader):
            batch = batch.to(self.device)
            h = global_mean_pool(self.model(batch), batch.batch)
            outputs = self.output_layer(h)
            targets = batch.y.view(outputs.shape).to(torch.float64)

            self.model_optimizer.zero_grad()
            is_valid = targets ** 2 > 0
            loss_mat = self.criterion(outputs.double(), (targets + 1) / 2)
            loss_mat = torch.where(
                is_valid, loss_mat,
                torch.zeros(loss_mat.shape).to(self.device).to(loss_mat.dtype))
            loss = torch.sum(loss_mat) / torch.sum(is_valid)
            loss = loss.mean()
            batch_num = targets.shape[0]

            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                self.norm_matrix[i * self.args.batch_size:min((i + 1) * self.args.batch_size, sample_num),
                self.cur_repeat] = torch.norm(torch.cat([bias_parameters_grads, (
                        h.view(batch_num, 1, embedding_dim).repeat(1,
                                             self.num_classes, 1) * bias_parameters_grads.view(
                                             batch_num, self.num_classes, 1).repeat(1, 1, embedding_dim)).
                                             view(batch_num, -1)], dim=1), dim=1, p=2)

        self.model.train()

    def select(self, **kwargs):
        # Initialize a matrix to save norms of each sample on idependent runs
        self.norm_matrix = torch.zeros([self.n_train, self.repeat], requires_grad=False).to(self.device)

        for self.cur_repeat in range(self.repeat):
            self.run()
            self.random_seed = self.random_seed + 5

        self.norm_mean = torch.mean(self.norm_matrix, dim=1).cpu().detach().numpy()
        top_examples = np.argsort(self.norm_mean)[::-1][:self.coreset_size]

        return {"indices": top_examples, "scores": self.norm_mean}
