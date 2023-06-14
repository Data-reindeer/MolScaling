from .earlytrain import EarlyTrain
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import pdb


class Uncertainty(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, num_tasks=None, epochs=200, selection_method="Entropy", 
                balance=False, **kwargs):
        # pdb.set_trace()
        super().__init__(dst_train, args, fraction, random_seed, num_tasks, epochs, **kwargs)

        selection_choices = ["LeastConfidence",
                             "Entropy",
                             "Margin"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method

        self.epochs = epochs
        self.balance = balance
        self.balance = 50

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_idx, epoch):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def finish_run(self):
        scores = self.rank_uncertainty()
        selection_result = np.argsort(scores)[::-1][:self.coreset_size]
        return {"indices": selection_result, "scores": scores}

    def rank_uncertainty(self, index=None):
        self.model.eval()
        with torch.no_grad():
            train_loader = DataLoader(self.dst_train, batch_size=self.args.batch_size,
                              shuffle=True, num_workers=self.args.num_workers)

            scores = np.array([])
            batch_num = len(train_loader)

            for i, batch in enumerate(train_loader):
                batch=batch.to(self.device)
                h= global_mean_pool(self.model(batch), batch.batch)
                outputs = self.output_layer(h)
                if i % self.print_freq == 0:
                    print("| Selecting for batch [%3d/%3d]" % (i + 1, batch_num))
                if self.selection_method == "LeastConfidence":                       
                    p = torch.sigmoid(outputs)
                    p[p<0.5] = 1 - p[p<0.5]                  
                    confidence = (1 - p).cpu().numpy()
                    scores = np.append(scores, confidence.sum(axis=1))

                elif self.selection_method == "Entropy":
                    p = torch.sigmoid(outputs)
                    entropy = (-p * torch.log(p + 1e-6) - (1-p) * torch.log(1 - p + 1e-6)).cpu().numpy()
                    scores = np.append(scores, entropy.sum(axis=1))

                elif self.selection_method == 'Margin':
                    preds = torch.nn.functional.softmax(outputs, dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    scores = np.append(scores, (max_preds - preds[
                        torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]).cpu().numpy())
        return scores

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
