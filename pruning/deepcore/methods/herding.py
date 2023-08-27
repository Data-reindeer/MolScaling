from .earlytrain import EarlyTrain
import torch
import numpy as np
from .methods_utils import euclidean_dist
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
from tqdm import tqdm
import pdb


class Herding(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200,
                balance: bool = False, metric="euclidean", **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs=epochs, **kwargs)

        if metric == "euclidean":
            self.metric = euclidean_dist
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist
            self.run = lambda: self.finish_run()

            def _construct_matrix(index=None):
                data_loader = DataLoader(self.dst_train, batch_size=self.args.batch_size,
                            shuffle=False, num_workers=self.args.num_workers)
                batch = next(iter(data_loader))
                inputs = batch.x
                return inputs.flatten(1).requires_grad_(False).to(self.device)

            self.construct_matrix = _construct_matrix

        self.balance = balance

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def construct_matrix(self, index=None):
        self.model.eval()
        self.output_layer.eval()
        sample_num = self.n_train if index is None else len(index)
        matrix = torch.zeros([sample_num, self.args.emb_dim], requires_grad=False).to(self.device)
        with torch.no_grad():
            data_loader = DataLoader(self.dst_train, batch_size=self.args.batch_size,
                            shuffle=False, num_workers=0)

            for i, batch in enumerate(data_loader):
                batch=batch.to(self.device)
                h= global_mean_pool(self.model(batch), batch.batch)
                matrix[i * self.args.batch_size:min((i + 1) * self.args.batch_size, sample_num)] = h

        # 这里的matrix即为完整数据的representation
        return matrix

    def herding(self, matrix, budget: int, index=None):
        sample_num = matrix.shape[0]

        if budget < 0:
            raise ValueError("Illegal budget size.")
        elif budget > sample_num:
            budget = sample_num

        indices = np.arange(sample_num)
        with torch.no_grad():
            mu = torch.mean(matrix, dim=0)
            select_result = np.zeros(sample_num, dtype=bool)

            for i in tqdm(range(budget)):
                if i % self.print_freq == 0:
                    print("| Selecting [%3d/%3d]" % (i + 1, budget))
                dist = self.metric(((i + 1) * mu - torch.sum(matrix[select_result], dim=0)).view(1, -1),
                                   matrix[~select_result])
                p = torch.argmax(dist).item()
                p = indices[~select_result][p]
                select_result[p] = True
        if index is None:
            index = indices
        return index[select_result]

    def finish_run(self):
        selection_result = self.herding(self.construct_matrix(), budget=self.coreset_size)
        return {"indices": selection_result}

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result

