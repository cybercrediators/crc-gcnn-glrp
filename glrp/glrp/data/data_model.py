import pprint
import pandas as pd
import numpy as np
from glrp.data.ge_dataset import GeDataset
from spektral.transforms import LayerPreprocess
from spektral.layers.convolutional import ChebConv
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.utils.convolution import chebyshev_filter, chebyshev_polynomial

from lib import coarsening

from scipy.sparse import csr_matrix

class DataModel:
    '''Define a data model and helper functions for the spektral dataset'''

    def __init__(self, config):
        self.config = config
        self.X = None
        self.Y = None
        self.A = None
        #self.X_train = None
        #self.X_test = None
        #self.Y_train = None
        #self.Y_test = None
        # patient indices for train/test subsets
        self.PI_train = None
        self.PI_test = None
        self.feature_values = None
        self.feature_graph = None
        self.labels = None
        if "precision" in self.config:
            self.precision = self.config["precision"]

        self.train_data = None
        self.test_data = None
        self.val_data = None

        # spektral dataset
        self.dataset = None
        #self.init_data()

        self.perm = None
        self.adj_coarsened = None

    def init_data(self):
        """
        Initialize the data from the given directories from
        the config file.
        """
        # TODO: set decay_steps in config correctly
        self.labels = pd.read_csv(self.config['path_to_labels'], dtype=self.precision)
        self.feature_values = pd.read_csv(self.config['path_to_feature_val'])
        self.feature_graph = pd.read_csv(self.config['path_to_feature_graph'], dtype=self.precision)
        #print(self.feature_values)
        #print(self.feature_values.shape)
        #print(self.feature_graph)
        self.coarsen_inputs(len(self.config["F"]), csr_matrix(self.feature_graph))

        # spektral dataset: graph obj. container
        self.dataset = GeDataset(self.labels, self.feature_values, self.feature_graph, self.config["normalize"], self.perm)
        #self.dataset.a = self.dataset.feature_graph
        #self.dataset.a = csr_matrix(self.dataset.feature_graph) 
        self.dataset.a = ChebConv.preprocess(csr_matrix(self.dataset.feature_graph))
        #self.dataset.a = sp_matrix_to_sp_tensor(self.dataset.a)

    def get_all_gene_names(self):
        """
        Retrieve and return all gene names from the features.
        """
        return self.feature_graph.columns.to_list()

    def coarsen_inputs(self, num_layers, adj_mat):
        """
        Preprocess inputs and coarsen the graphs (according
        to the original proposal)
        """
        graphs, self.perm = coarsening.coarsen(adj_mat, levels=num_layers, self_connections=False)
        self.adj_coarsened = [ChebConv.preprocess(a) for a in graphs]
        #self.adj_coarsened = [sp_matrix_to_sp_tensor(x for x in self.adj_coarsened]

    def train_test_split(self, size, predefined_test_patients=None, seed=None):
        '''create a train and test split from input data'''
        print(self.dataset)
        # split the dataset by the predefined test patients
        rng = np.random.default_rng()
        if seed:
            rng = np.random.default_rng(seed=seed)
        if predefined_test_patients:
            # get patient ids
            ids = np.arange(len(self.dataset))
            patients = self.feature_values.columns.to_list()[:-1]
            #print(patients)
            #for idx, pat in enumerate(patients):
            #    print(idx, pat)
            self.PI_test = [i for i, x in enumerate(patients) if x in predefined_test_patients]
            #print(self.PI_test)
            self.PI_train = [x for x in ids if x not in self.PI_test]
        else:
            # create train/test split
            #idx = np.random.permutation(len(self.dataset))
            idx = rng.permutation(len(self.dataset))
            test_split = int((1 - size) * len(self.dataset))

            # get patient ids which got in train/test sets respectively
            self.PI_train, self.PI_test = np.split(idx, [test_split])

        self.train_data = self.dataset[self.PI_train]
        self.test_data = self.dataset[self.PI_test]
        #print(self.PI_test[0])

        #print(self.test_data[0].y)
        #print(self.test_data[0].x)

        print("Train split: ", self.train_data)
        print("Test split:", self.test_data)

    def train_test_val_split(self, test_size, val_size):
        '''create a train, validation and test split from input data'''
        print(self.dataset)
        # create train/test split
        idx = np.random.permutation(len(self.dataset))
        test_split = int((1 - size) * len(self.dataset))

        # get patient ids which got in train/test sets respectively
        self.PI_train, self.PI_test = np.split(idx, [test_split])

        self.train_data = self.dataset[self.PI_train]
        self.test_data = self.dataset[self.PI_test]

        print("Train split: ", self.train_data)
        print("Test split:", self.test_data)

    def show_data_infos(self):
        '''Show information about the processed data from this data class'''
        pp = pprint.PrettyPrinter(indent=4)
        print("Feature Values: ", self.feature_values.shape)
        print("Graph: ", self.feature_graph.shape)
        print("Labels: ", self.labels.shape)
