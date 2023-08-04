Transductive learning
The input to the transductive model contains:

x, the feature vectors of the training instances,
y, the one-hot labels of the training instances,
graph, a dict in the format {index: [index_of_neighbor_nodes]}, where the neighbor nodes are organized as a list. The current version only supports binary graphs.
Let L be the number of training instances. The indices in graph from 0 to L - 1 must correspond to the training instances, with the same order as in x.

Inductive learning
The input to the inductive model contains:

x, the feature vectors of the labeled training instances,
y, the one-hot labels of the labeled training instances,
allx, the feature vectors of both labeled and unlabeled training instances (a superset of x),
graph, a dict in the format {index: [index_of_neighbor_nodes]}.
Let n be the number of both labeled and unlabeled training instances. These n instances should be indexed from 0 to n - 1 in graph with the same order as in allx.

In addition to x, y, allx, and graph as described above, the preprocessed datasets also include:

tx, the feature vectors of the test instances,
ty, the one-hot labels of the test instances,
test.index, the indices of test instances in graph, for the inductive setting,
ally, the labels for instances in allx.
The indices of test instances in graph for the transductive setting are from #x to #x + #tx - 1, with the same order as in tx.

You can use cPickle.load(open(filename)) to load the numpy/scipy objects x, y, tx, ty, allx, ally, and graph. test.index is stored as a text file.