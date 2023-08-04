"""
    Obtain a message reinforcement matrix based on the ParWalks random walking model
    according to different given parameters(alpha,beta)
"""
from utils.DataStorage import save_sparse_matrix
from utils.ParWalksEnhancedMatrixUitls import getParWalksProbMatrix
from utils.enhanced_utilities import getEnhancedMatrix

dataset_str = 'cora'
P = getParWalksProbMatrix(dataset_str=dataset_str)
beta_list = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20]
alpha_list = [99.9, 99.8, 99.7, 99.6, 99.5, 99.4, 99.3, 99.2, 99.1, 99.0]
dir = 'ParWalksEnhancedMatrixForCora'
suffix = '.npz'
for i in range(len(alpha_list)):
    upper_percent = alpha_list[i]
    for j in range(len(beta_list)):
        beta = beta_list[j]
        parwalks_enahnced_matrix = getEnhancedMatrix(probs=P, upper_percent=upper_percent, beta=beta)
        fileName = dir + '/' + 'parwalks_enhanced_upper_percent_' + str(upper_percent) + '%_beta_' + str(beta) + suffix
        save_sparse_matrix(fileName, parwalks_enahnced_matrix)
        print(str(upper_percent) + '_' + str(beta) + '_' + str(parwalks_enahnced_matrix.nnz))

print('Done')
