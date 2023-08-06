# GCN-MF

This repository is the  implementation of  GCN-MF.

## Overview

In this project, we provide implementations of our model GCN-MF, and compare it with GCN and GAT. The repository is organised as follows:

- `data/` contains the benchmark datasets: Cora and CiteSeer.
- `FilterMatrix/` contains the filter matrix for Cora and CiteSeer.
- `ParameterAnalysisForCora/` contains the analysis of the influences of filter parameters on GCN-MF performance for Cora.
- `splits/` contains different splitting of train set,validation set and test set on dataset Cora and CiteSeer.
- `utils/` contains common functions such as loading dataset, saving matrix and so on,also including the implementations of GCN and GCN-MF.
- `GCN-MF.py` contains the training process of GCN and GCN-MF.
- `GeneratingDifferentFilterMatrix.py` contains the process of generating filter matrix.
- `ParamAnalysisWithTraining.py` aims to get different training results (accuracies) using filter matrix.
- `PlotParamAnalysis.py` aims to plot a 3d figure to reveal the influences of filter parameters on classification accuracy.
- `Param_accuracy.txt` aims to store the training accuracies with respective to filters with different parameters.


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

Run commands below to get the 3d figure that reveals the influence of filter parameters. 

```visual
python PlotParamAnalysis.py
```

## Models

To train the GCN-MF model, run the  respective command below:

```train models
python GCN-MF.py
```

By specifying arguments in `GCN-MF.py`, you can change the dataset and adjust hyper-parameters.
The hyper-parameters we have used are shown below

- model layers = 4
- weight decay = 1e-4
- learning rate = 0.01
- dropout rate = 0.0

## Datasets

In this paper, we use two citation datasets: Cora and CiteSeer. They are citation graphs, where a node represents a paper, and an edge between two nodes represents that the two papers have a citation relationship.
