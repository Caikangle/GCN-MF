import numpy as np

import scipy.sparse as sp


# Calculate the corresponding percentile according to the passed-in parameters
def getCentileValue(probs, upper_percent=90):
    a = []
    for i in range(probs.shape[0]):
        for j in range(i + 1, probs.shape[0]):
            if probs[i, j] != 0:
                a.append(probs[i, j])
    upper_bound_value = np.percentile(a, upper_percent)
    return upper_bound_value


# get the message reinforcement matrix
def getEnhancedMatrix(probs, upper_percent=90, beta=0.1):
    upper_bound_value = getCentileValue(probs, upper_percent)
    mask = sp.csr_matrix(np.zeros((probs.shape[0], probs.shape[1])))
    for i in range(probs.shape[0]):
        for j in range(i + 1, probs.shape[0]):
            if probs[i, j] >= upper_bound_value:
                mask[i, i] = 1 - beta
                mask[i, j] = beta
                mask[j, j] = 1 - beta
                mask[j, i] = beta
    return mask
