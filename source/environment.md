## Prepare Environment

### Requirements

- [Python](http://www.python.org/) (>=3.7)
- [PyTorch](https://pytorch.org/docs/stable/index.html) (==1.12.1)
- [PyG](https://pytorch-geometric.readthedocs.io/) (==1.10.2)
- [Transformers](https://huggingface.co/docs/transformers) (==4.17.0)
- [RDkit](https://www.rdkit.org/) (==2020.09.1.0)
- [ASE](https://wiki.fysik.dtu.dk/ase/index.html) (==3.22.1)
- [DescriptaStorus](https://github.com/bp-kelley/descriptastorus) (==2.3.0.5)
- [OGB](https://ogb.stanford.edu/docs/home/) (==1.3.3)



### Installation

We recommend users to create a virtual environment using [Anaconda](https://www.anaconda.com/). Below, we provide the steps for creating the environment and installing the required packages. The txt file containing the requirements has been uploaded to the GitHub repository.

```shell
conda create --name NSL_MRL python=3.7.11
conda activate NSL_MRL
```

```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.10.2%2Bcu113.html

pip install -r requirements.txt
```

```shell
git clone git@github.com:Data-reindeer/NSL_MRL.git
```



### Dataset Preparation

**Datasets**

We offer the processed dataset on [Google Drive](https://drive.google.com/drive/folders/1sWrG8ZhBvx9lrfzMHEhbEpLPjHuBdjm_?usp=drive_link). To access the datasets, please download and extract the *HIV*, *MUV*, and *PCBA* datasets from the `molecule_net.zip` file, while the *QM9* dataset can be extracted from `qm9.zip`. Please place the extracted files in the `./datasets/` folder, maintaining the following hierarchical structure:

```shell
./datasets/molecule_net
./datasets/qm9
```

