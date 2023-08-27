# Model Classes

We utilize different models to encode distinct molecular modalities. In our paper, the adopted models are well-established and widely recognized encoder structures in their field. Below is an introduction to these model classes for the four modalities: 2D graphs, 3D graphs, morgan fingerprints, and SMILES strings.



### 2D Graph

For the 2D graph modality, we utilize the [Graph Isomorphism Network (GIN)](https://arxiv.org/abs/1810.00826) as the encoder. 

>  *Class* **GNN** (num_layer, emb_dim, JK, drop_ratio, gnn_type)

**PARAMETERS**

- `num_layers`(int): The number of GNN layers.  
- `emb_dim` (int): dimensionality of embeddings.
- `JK` (str): The Jumping Knowledge mode. If specified, the model will additionally apply a final linear transformation to transform node embeddings to the expected output feature dimensionality. (`last`, `concat`, `max` or `sum`) (default: `None`)
- `drop_ratio` (float): Dropout probability. (default: `0`)
- `gnn_type` (str): GNN type to use. (`gin`, `gcn`, `gat`). (defauly: `gin`)

```python
def forward(self, z, pos):
```

**PARAMETERS**

- `z`: atom type matrix with shape `[num_nodes]`.
- `pos`: atom Cartesian coordinates with shape `[num_nodes, 3]`.



### 3D Graph

For the 3D geometry modality, we employ the classical [SchNet](https://proceedings.neurips.cc/paper/2017/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf) model as the encoder.

> *Class* **SchNet** (hidden_channels, num_filters, num_interactions, num_gaussians, cutoff, readout)

**PARAMETERS**

- `hidden_channels` (int): Number of features to describe atomic environments.
- `num_filters` (int): Number of filters used in continuous-filter convolution
- `num_interactions` (int): Number of interaction blocks.
- `num_gaussians` (int): Number of Gaussian functions used to model atom distances.
- `cutoff` (float): Distance beyond which interactions are truncated to reduce complexity.
- `readout` (str): Readout function to extract molecular output.

```python
def forward(self, x, edge_attr, edge_index):
```

**PARAMETERS**

- `x`: Node feature matrix with shape `[num_nodes, num_node_features]`;

- `edge_attr`: edge feature matrix with shape `[num_edges, num_edge_features]`;
- `edge_index`: Graph connectivity in [COO format](https://pytorch.org/docs/stable/sparse.html#sparse-coo-docs) with shape `[2, num_edges]` and type `torch.long`;



### Morgan Fingerprint and SMILES String

For the fingerprint modality, we use RDKit to generate 1024-bit molecular fingerprints with radius R = 2, which is roughly equivalent to the ECFP4 scheme. We adopt Transformer to encode the fingerprint. For the SMILES modality, we employ the same model architecture as the fingerprint modality to ensure a fair comparison.

> *Class* **Transformer** (word_dim, out_dim, num_head, num_layer)

**PARAMETERS**

- `word_dim` (int): The dimensionality of the word embeddings or input embeddings. It specifies how many features are used to represent each word in the input sequence.
- `out_dim` (int): The output and hidden dimensionality of the Transformer model.
- `num_head` (int): The number of attention heads in the multi-head self-attention mechanism.
- `num_layer` (int): The number of stacked transformer layers.

```python
def forward(self, fingerprint):
```

**PARAMETERS**

- `fingerprint`: bit vector matrix with shape`[num_bits]`.



Users have the flexibility to incorporate their own models for testing as needed, with the only requirement being that the input format of the `forward` function of the custom model should align with the corresponding modality.