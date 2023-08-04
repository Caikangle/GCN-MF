"""
    Obtain the probability matrix based on ParkWalks
"""
import os

import numpy as np
from numpy.linalg import inv

from utils.preprocess_help import load_data


def getParWalksProbMatrix(dataset_str, alpha=1e-6):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    adj, _, _, _, _, _ = load_data(dataset_str)
    A_tilde = adj.toarray() + np.identity(adj.shape[0])
    D = A_tilde.sum(axis=1)
    A_ = np.diag(D ** -0.5).dot(A_tilde).dot(np.diag(D ** -0.5))

    Lambda = np.identity(len(A_))
    L = np.diag(D) - adj
    P = inv(L + alpha * Lambda)
    return P
