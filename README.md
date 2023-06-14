# Neural Scaling Law in Molecular Representation Learning

This repository provides the source code for the paper **Uncovering Neural Scaling Law in Molecular Representation Learning**. 

## Environments

```markdown
numpy             1.21.2
scikit-learn      1.0.2
pandas            1.3.4
python            3.7.11
torch             1.10.2+cu113
torch-geometric   2.0.3
transformers      4.17.0
rdkit             2020.09.1.0
ase               3.22.1
descriptastorus   2.3.0.5
ogb               1.3.3
```

## Dataset Preparation

### Datasets

We offer the processed dataset on Google Drive: [https://drive.google.com/drive/folders/1sWrG8ZhBvx9lrfzMHEhbEpLPjHuBdjm_?usp=drive_link](https://drive.google.com/drive/folders/1sWrG8ZhBvx9lrfzMHEhbEpLPjHuBdjm_?usp=drive_link). YTo access the datasets, please download and extract the HIV, MUV, and PCBA datasets from the molecule_net.zip file, while the QM9 dataset can be extracted from qm9.zip. Please place the extracted files in the ./datasets/ folder, maintaining the following hierarchical structure:

```bash
./datasets/molecule_net
./datasets/qm9
```

## Experiments

We investigate the neural scaling behaviors of molecular representation learning (**MRL**) from a data-centric perspective across various dimensions，including (1) *data modality*, (2) *data distribution*, (3) *pre-training intervention*, and (4) *model capacity*. 

Apart from empirical discovery, we further adapt seven popular data pruning strategies to molecular data to seek the possibility to beat the scaling law. Below, we will provide detailed command with reproducible hyperparameter settings for each dimension. 

- **General Neural Scaling law**

Please use the default settings in config.py for unspecified hyperparameters.

```bash
# finetue_ratio: [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8]
# property:      [gap, U0, zpve]
# dataset:       [hiv, muv, pcba]
python main_graph.py --finetune_ratio=0.1 --dataset=hiv
python main_3d.py --finetune_ratio=0.1 --property=gap
```

- **The Effect of Molecular Modality**

```bash
python main_graph.py --finetune_ratio=0.1 --dataset=hiv
python main_fingerprint.py --finetune_ratio=0.1 --dataset=hiv
python main_smiles.py --finetune_ratio=0.1 --dataset=hiv
```

- **The Effect of Pre-training**

We offer the pretrained weight of GIN model in ./model_saved. However, you also have the option to retrain the pre-trained model from scratch.

```bash
python main_graph.py --finetune_ratio=0.1 --dataset=hiv --pretrain
```

- **The Effect of Data Distribution**

```bash
# split:      [random, scaffold, imbalanced]
python main_graph.py --finetune_ratio=0.1 --dataset=hiv --split=random
```

- **Data Pruning**

Note that Uncertainty method includes two measurements: Entropy and Least Confidence. If you choose other pruning methods, please ignore the Uncertainty configuration below.

```bash
# selection:      [Herding, Uncertainty, Forgetting, GraNd, Kmeans]
# Uncertainty:      [Entropy, LeastConfidence]
python main_graph.py --finetune_ratio=0.1 --dataset=hiv --finetune_pruning --selection=Kmeans --uncertainty=Entropy
```
