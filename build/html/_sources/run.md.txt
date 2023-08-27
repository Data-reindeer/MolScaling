# Experiments

Users can utilize the following commands to reproduce the results presented in our paper or to explore neural scaling laws of their custom settings.

- **General Neural Scaling law**

Please use the default settings in `config.py` for unspecified hyperparameters.

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

We offer the pretrained weight of GIN model in` ./model_saved`. However, you also have the option to retrain the pre-trained model from scratch.

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
