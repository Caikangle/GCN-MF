# GCN-MF

This repository is the  implementation of  GCN-MF.

## Overview

In this project, we provide implementations of two models -- GCN, GCN-MF and parameter analysis on the model GCN-NF. About the construction and  analysis of the model GAT, we implement multi-layer GAT models using the library PyG(PyTorch Geometric). The repository is organised as follows:

- `data/` contains the necessary dataset files for Cora, CiteSeer.
- `ParWalksEnhancedMatrix/` contains the message reinforcement matrix of dataset Cora and CiteSeer.
- `ParWalksEnhancedMatrixForCora/` contains the different message reinforcement matrices of dataset Cora ,aiming to analyse parameters of GCN-MF.
- `splits/` contains different spliting of train set,validation set and test set on dataset Cora,CiteSeer.
- `utils` contains common functions such as loading dataset, saving matrix and so on,also including the implementations of GCN and GCN-MF.
- `GCN-MF.py` contains the training process of GCN and GCN-MF.
- `GeneratingDifferentEnhancedMatrix.py` contains the process of generating different message reinforcement matrices.
- `ParamAnalysisWithTraining.py` aims to get different training results (accuracies) according to different message reinforcement matrices.
- `PlotParamAnalysis.py` aims to plot a 3d figure to analyse the impact of different enhanced matrices on  the final accuracy.
- `alpha_beta_accuracy.txt` aims to store the training results (accuracies) according to different message reinforcement matrices.


## Requirements

The project runs under Python 3.6.8 with main required dependencies:

* numpy==1.19.4
* scipy==1.5.4
* matplotlib==3.3.4
* networkx==2.5
* torch==1.8.0+cu111
* scikit_learn==1.3.0

More detailed information about dependencies, you can get in `requirements.txt`.

In addition, CUDA 11.1 is used in this project.

## Visual analysis

Run commands below to get the 3d figure of analysing enhanced matrices . 

```visual
python PlotParamAnalysis.py
```

## Models

To train the GCN-MF model, run the  respective command below:

```train models
python GCN-MF.py
```

By specifying arguments in 'GCN-MF.py', you can change the dataset and adjust hyper-parameters.
The hyper-parameters we have used are shown below

- model layers = 4
- weight decay = 1e-4
- learning rate = 0.01
- dropout rate = 0.0

## Datasets

In this paper, we use two citation datasets.
Cora and CiteSeer are citation graphs, where a node represents a paper, and an edge between two nodes represents that the two papers have a citation relationship.
