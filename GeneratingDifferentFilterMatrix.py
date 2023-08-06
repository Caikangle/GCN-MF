"""
    Generate different filter matrices
"""
from utils.DataStorage import save_sparse_matrix
from utils.FilterMatrixUitls import getParWalksProbMatrix
from utils.enhanced_utilities import getFilterMatrix

dataset_str = 'cora'
P = getParWalksProbMatrix(dataset_str=dataset_str)
beta_list = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.20]
upper_percent_list = [99.9, 99.8, 99.7, 99.6, 99.5, 99.4, 99.3, 99.2, 99.1, 99.0]
dir = 'ParameterAnalysisForCora'
suffix = '.npz'
for i in range(len(upper_percent_list)):
    upper_percent = upper_percent_list[i]
    for j in range(len(beta_list)):
        beta = beta_list[j]
        parwalks_filter_matrix = getFilterMatrix(probs=P, upper_percent=upper_percent, beta=beta)
        fileName = dir + '/' + 'parwalks_filter_matrix_upper_percent_' + str(upper_percent) + '%_beta_' + str(beta) + suffix
        save_sparse_matrix(fileName, parwalks_filter_matrix)
        print(str(upper_percent) + '_' + str(beta) + '_' + str(parwalks_filter_matrix.nnz))

print('Done')
