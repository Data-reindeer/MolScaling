from .earlytrain import EarlyTrain
import torch, time
import math
import numpy as np
import pdb
from sklearn.metrics import roc_auc_score, average_precision_score


# Acknowledgement to
# https://github.com/mtoneva/example_forgetting

class Forgetting(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, num_tasks=None, 
                 epochs=200, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, num_tasks, epochs, **kwargs)

    def get_hms(self, seconds):
        # Format time for printing purposes

        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)

        return h, m, s

    def before_train(self):
        self.train_loss = 0.
        self.correct = 0.
        self.total = 0.

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        with torch.no_grad():
            preds = torch.sigmoid(outputs)
            preds[preds<0.5] = -1
            preds[preds>=0.5] = 1
            cur_acc = (preds == targets).sum(axis=1)
            cur_acc = cur_acc.clone().detach().requires_grad_(False).type(torch.float32)
            self.forgetting_events[torch.tensor(batch_inds)[(self.last_acc[batch_inds]-cur_acc)>0.01]] += 1.
            self.last_acc[batch_inds] = cur_acc

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        self.train_loss += loss.item()
        self.total += targets.size(0)*self.num_classes
        preds = torch.sigmoid(outputs)
        preds[preds<0.5] = -1
        preds[preds>=0.5] = 1
        self.correct += preds.eq(targets).cpu().sum()

        if batch_idx % self.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' % (
            epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item(),
            100. * self.correct.item() / self.total))

    def before_epoch(self):
        self.start_time = time.time()

    def after_epoch(self):
        epoch_time = time.time() - self.start_time
        self.elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (self.get_hms(self.elapsed_time)))

    def before_run(self):
        self.elapsed_time = 0

        self.forgetting_events = torch.zeros(self.n_train, requires_grad=False).to(self.device)
        self.last_acc = torch.zeros(self.n_train, requires_grad=False).to(self.device)

    def finish_run(self):
        pass

    def select(self, **kwargs):
        self.run()
        top_examples = self.train_indx[np.argsort(self.forgetting_events.cpu().numpy())][::-1][:self.coreset_size]

        return {"indices": top_examples, "scores": self.forgetting_events}
