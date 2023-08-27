import argparse

parser = argparse.ArgumentParser()
# ========= Seed and basic info ==========
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--runseed', type=int, default=1)
parser.add_argument('--device', type=int, default=5)

# ========= Hyper-parameters ===========
parser.add_argument('--dataset', type=str, default='pcba')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--decay', type=float, default=0)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--lr_decay_step_size', type=int, default=150)
parser.add_argument('--lr_decay_factor', type=int, default=0.5)
parser.add_argument('--lr_cosine_length', type=int, default=400000, help='Cosine length if lr_schedule is cosine.')
parser.add_argument('--lr_warmup_steps', type=int, default=1e4, help='Warm-up Steps.')
parser.add_argument('--patience', type=int, default=20, help='Early stopping patiance.')
parser.add_argument('--decay_patience', type=int, default=5, help='Scheduler decay patiance.')
parser.add_argument('--decay_factor', type=float, default=0.5, help='Scheduler decay patiance.')
parser.add_argument('--mask_rate', type=int, default=0.15)


# ======== Model configuration =========
parser.add_argument('--net2d', type=str, default='GIN')
parser.add_argument('--net_sm', type=str, default='transformer')
parser.add_argument('--num_layer', type=int, default=5)
parser.add_argument('--emb_dim', type=int, default=300)
parser.add_argument('--dropout_ratio', type=float, default=0.5) 
parser.add_argument('--graph_pooling', type=str, default='mean')
parser.add_argument('--JK', type=str, default='last')
parser.add_argument('--output_model_dir', type=str, default='./model_saved/')
parser.add_argument('--property', type=str, default='lumo', help='Regression Target')

# ========= Program viewing =========== 
parser.add_argument('--eval_train', dest='eval_train', action='store_true')
parser.add_argument('--no_eval_train', dest='eval_train', action='store_false')
parser.set_defaults(eval_train=False)

# ========= Neural Scaling Parameter =======
parser.add_argument('--selection', type=str, default='Uncertainty')
parser.add_argument('--selection_epochs', type=int, default=20)
parser.add_argument('--selection_lr', type=float, default=1e-3)
parser.add_argument('--selection_decay', type=float, default=0)
parser.add_argument('--split', type=str, default='random')
parser.add_argument('--finetune_pruning', action='store_true')
parser.add_argument('--finetune_ratio', type=float, default=0.1)
parser.add_argument('--K', type=int, default=100)
parser.add_argument('--uncertainty', default="Entropy", help="specifiy uncertanty score to use")
parser.add_argument('--pretrain', action='store_true')
parser.set_defaults(finetune_pruning=False)
parser.set_defaults(pretrain=False)

args = parser.parse_args()
print('arguments\t', args)

