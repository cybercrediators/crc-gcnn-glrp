import numpy as np
from spektral.data import Dataset, Graph
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from sklearn.preprocessing import StandardScaler

from lib import coarsening


class GeDataset(Dataset):
    """
    Dataset for crc data
    """
    def __init__(self, labels, feature_vals, feature_graph, normalize, perm=None, **kwargs):
        self.labels = labels
        self.feature_vals = feature_vals
        self.feature_graph = feature_graph
        self.n_samples = 0
        self.normalize = normalize
        self.perm = perm
        super().__init__(**kwargs)

    def read(self):
        '''Read the input features and feature graph and return a spektral dataset'''
        graphs = []

        # TODO: autosave/load graph data from disk

        self.labels = self.labels.to_numpy()[0].astype('float32')
        self.feature_graph = csr_matrix(self.feature_graph.values)
        self.feature_vals = np.delete(self.feature_vals.to_numpy(), -1, axis=1)
        self.feature_vals = np.asarray(self.feature_vals).astype('float32')

        self.n_samples = self.labels.shape[0]

        # normalize inputs
        if self.normalize:
            #self.feature_vals = (self.feature_vals - np.min(self.feature_vals)) / np.ptp(self.feature_vals)
            #self.feature_vals = StandardScaler().fit_transform(self.feature_vals)
            #print(self.feature_vals)
            self.feature_vals = self.feature_vals - np.min(self.feature_vals)
        self.feature_vals = self.feature_vals.T
        if self.perm:
            self.feature_vals = coarsening.perm_data(self.feature_vals, self.perm)

        # init graph data
        # spektral graph obj.:
        #   - x: node features
        #   - a: adj. matrix
        #   - y: graph/node labels
        for i in range(self.n_samples):
            x = self.feature_vals[i,:]
            x = x.reshape(x.shape[0], 1)
            y = self.labels[i]
            #graph = Graph(x=x, a=self.feature_graph, y=y)
            graph = Graph(x=x, a=None, y=y)
            graphs.append(graph)
        return graphs
