import argparse
from ast import Store
from email.policy import default

def parse_args():
    parser = argparse.ArgumentParser(description='')

    # dataset: weibo, twitter15, twitter16
    TDdroprate=0.2
    BUdroprate=0.2

    # dataset realated
    parser.add_argument("--dataset", type=str, choices=['Weibo', 'Twitter15', 'Twitter16'], help="dataset")

    # train related
    parser.add_argument("--lr", type=float, default=0.0005, help='learning rate')
    parser.add_argument("--weight_decay", type=float, default=1e-4, help='weight decay rate')
    parser.add_argument("--patience", type=int, default=10, help='early stopping patience')
    parser.add_argument("--n_epochs", type=int, default=200, help="num of epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--iteration", type=int, default=10, help="iteration num")

    # model
    parser.add_argument("--model", type=str, default="gt", choices=["gt", "bigcn"])
    parser.add_argument("--dim_in", type=int, default=5000, help="")
    parser.add_argument("--dim_hidden", type=int, default=128, help="")
    parser.add_argument("--dim_out", type=int, default=128, help="")
    parser.add_argument("--k_hop", type=int, default=2, help="")
    parser.add_argument("--gt_layer_type", type=str, default="GCN+Transformer", help="")
    parser.add_argument("--gt_layers", type=int, default=1, help="num of gt_layers")
    parser.add_argument("--gt_n_heads", type=int, default=4)
    parser.add_argument("--gt_pna_degrees", default=None,)
    parser.add_argument("--posenc_EquivStableLapPE_enable", action="store_true", help="")       # default=False
    parser.add_argument("--gt_dropout", type=float, default=0.1, help="")
    parser.add_argument("--gt_attn_dropout", type=float, default=0.1, help="")
    parser.add_argument("--gt_layer_norm", action="store_true", help="")                       # default=False
    parser.add_argument("--gt_batch_norm", action="store_false", help="")                       # default=True
    parser.add_argument("--gt_bigbird", default=None)

    parser.add_argument("--TDdroprate", type=float, default=0.0, help="")
    parser.add_argument("--BUdroprate", type=float, default=0.0, help="")

    return parser
