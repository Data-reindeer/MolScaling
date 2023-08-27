# Dataset Classes

For different molecular modalities, we utilize different classes to integrate necessary fields. Below is an introduction to the dataset classes for the four modalities: 2D graphs, 3D graphs, morgan fingerprints, and SMILES strings.



### 2D Graph and Morgan Fingerprint

For 2D graph and Morgan fingerprint modalities, we have MoleculeNet (HIV, MUV and PCBA) and PCQM4Mv2 datasets. The hierarchy of corresponding folders is as follows:

```
-- HIV
  -- raw
    -- hiv.csv

  -- processed
    -- geometric_data_processed.pt
    -- pre_filter.pt
    -- pre_transform.pt
    -- smiles.csv
```

We use one class to integrate 2D graph and Morgan Fingerprint modalities. The dataset classes `MoleculeDataset` and `PygPCQM4Mv2Dataset` are integrated by PyG's *[InMemoryDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.InMemoryDataset.html?highlight=InMemoryDataset)* class, with each data point containing the following fields:

- `data.x`: Node feature matrix with shape `[num_nodes, num_node_features]`;

- `data.edge_attr`: edge feature matrix with shape `[num_edges, num_edge_features]`;
- `data.edge_index`: Graph connectivity in [COO format](https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs) with shape `[2, num_edges]` and type `torch.long`;
- `data.fingerprint`: bit vector matrix with shape`[num_bits]`;
- `data.id`: unique identifier for the example;
- `data.y`: labels with shape `[num_tasks]`.



### 3D Graph

For 3D graph modality of QM9 datasets, the hierarchy of folders is as follows:

```
-- QM9
  -- raw
    -- qm9_eV.npz

  -- processed
    -- qm9_pyg.pt
    -- pre_filter.pt
    -- pre_transform.pt
```

The corresponding dataset class `QM93D` is also integrated by PyG's *[InMemoryDataset](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.InMemoryDataset.html?highlight=InMemoryDataset)* class, with each data point containing the following fields:

- `data.pos`: atom Cartesian coordinates with shape `[num_nodes, 3]`;

- `data.z`: atom type matrix with shape `[num_nodes]`;
- `data.y`: labels with shape `[num_tasks]`.



### SMILES String

For SMILES string modality, we integrated the samples using the [TensorDataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.TensorDataset) class from PyTorch, with each data point containing the following fields:

- `data.input_id`: Tokenized numerical representation of SMILES string with shape `[pad_length]`, which will be used as input by the model.
- `data.attention_mask`: A binary tensor with shape `[pad_length]`, indicating the position of the padded indices so that the model does not attend to them. 
- `data.y`: labels with shape `[num_tasks]`



If users intend to explore the neural scaling law using custom datasets, they should prepare and structure their datasets according to the modalities mentioned above for conducting experiments.
