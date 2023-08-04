"""
    save or load a sparse matrix
"""
from scipy import sparse


def load_sparse_matrix(path):
    """
    load a sparse matrix from path
    :param path:
    :return:
    """
    return sparse.load_npz(path)


def save_sparse_matrix(path, sparse_matrix):
    """
    save a sparse matrix to the path
    :param path:
    :param sparse_matrix:
    :return:
    """
    sparse.save_npz(path, sparse_matrix)
