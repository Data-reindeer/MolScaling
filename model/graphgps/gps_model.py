import torch
import torch_geometric.graphgym.register as register
import torch.nn as nn

from .gps_layer import GPSLayer

num_atom_type = 120  # including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6  # including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3

class GPSModel(torch.nn.Module):
    """Multi-scale graph x-former.
    """

    def __init__(self, dim_hidden, dim_out, args):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.gt_layers = args.gt_layers
        self.x_embedding1 = nn.Embedding(num_atom_type, dim_hidden)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, dim_hidden)

        # self.edge_embedding1 = nn.Embedding(num_bond_type, dim_hidden)
        # self.edge_embedding2 = nn.Embedding(num_bond_direction, dim_hidden)

        try:
            local_gnn_type, global_model_type = args.gt_layer_type.split('+')
        except:
            raise ValueError(f"Unexpected layer type: {args.gt_layer_type}")
        layers = []
        for _ in range(self.gt_layers):
            layers.append(GPSLayer(
                dim_h=self.dim_hidden,
                local_gnn_type=local_gnn_type,
                global_model_type=global_model_type,
                num_heads=args.gt_n_heads,
                equivstable_pe=args.posenc_EquivStableLapPE_enable,
                dropout=args.gt_dropout,
                attn_dropout=args.gt_attn_dropout,
                layer_norm=args.gt_layer_norm,
                batch_norm=args.gt_batch_norm,
                bigbird_cfg=args.gt_bigbird,
            ))
        self.layers = torch.nn.Sequential(*layers)


    def forward(self, batch):
        # batch.edge_attr = self.edge_embedding1(batch.edge_attr[:, 0]) + \
        #                   self.edge_embedding2(batch.edge_attr[:, 1])
        batch.x = self.x_embedding1(batch.x[:, 0]) + self.x_embedding2(batch.x[:, 1])
        for module in self.layers:
            batch = module(batch)
        return batch.x
