# Improving crystal material property prediction through multi-view graph transformer framework

![1.png]

## Installation

First install miniconda:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then set up environment and install packages:
```
conda create -n MGTransformer python==3.9
conda activate MGTransformer

# install pytorch and pytorch-geometric
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-2.5.1+cu124.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-2.5.1+cu124.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-2.5.1+cu124.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-2.5.1+cu124.html
pip install torch-geometric

# install other packages
pip install scipy==1.13.0
pip install e3nn==0.5.3
pip install ase==3.22.1
pip install jarvis-tools==2022.9.16
pip install pymatgen==2023.11.12
pip install transformers==4.43.4
pip install dgl==1.1.1

pip install tqdm==4.66.1
pip install scikit-learn==1.3.1
pip install seaborn==0.13.2
pip install pandas==2.2.3
```
This project uses CUDA 12.4. If using a different version, ensure your CUDA Toolkit, PyTorch, and PyTorch Geometric (PyG) are mutually compatible. For setup details, refer to the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html#).


## Dataset 

We have prepared the relevant processed datasets, which can be used directly for your convenience. Please download the pre-training, fine-tuning, and transfer datasets used in the paper [here](https://doi.org/10.5281/zenodo.15473642).

Once you have successfully downloaded the datasets, please follow these steps for organization:

#### Pretraining Datasets: 

Extract the pre-training dataset and unzip it under the `./dataset/pretrained` folder. Additionally, we have provided a pre-training debug dataset to assist you in debugging your code.

#### Fine-tuning and Transfer Learning Datasets:

Extract the fine-tuning and transfer learning datasets and unzip them under the `./dataset/fine-tuning` folder.

#### Process dataset:

If you prefer to handle each pre-training and fine-tuning dataset independently, we have provided relevant command lines and detailed instructions. You can find more information in the `./dataset/README.md` file.

## Pre-trained models

The pre-trained MGT for pretraining can be found in `ckpt/pretraining` folder. 

All downstream tasks of MGT for `tutorial.ipynb` can be found in `ckpt/finetuned` folder.

## Pretraining

To train the MGT framework, where the configurations and detailed explaination for each variable can be found in `config/pretraining.yml` folder.

```
python pretraining.py
```

## Fine-tuning 

To fine-tune the pre-trained model on downstream prediction tasks, where the configurations and detailed explaination for each variable can be found in `config/finetune.yml`

```
python finetune.py
```

## Interferce

A tutorial notebook for interferce process is available in `tutorial.ipynb`.

```
jupyter notebook tutorial.ipynb
```


## License

MGT is released under the [MIT](LICENSE) license.

## Contact

If you have any questions, please reach out to liang36365@mail.ustc.edu.cn
