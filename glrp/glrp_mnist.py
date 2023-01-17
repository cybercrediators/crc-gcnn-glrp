import tensorflow as tf
import numpy as np
import spektral
import pprint
from glrp.helpers import config, utils, storage
from glrp.trainer.gcnn_runner import GCNNRunner
from glrp.trainer.test_runner import TestingRunner
from glrp.trainer.mnist_runner import MnistRunner
from glrp.data.data_model import DataModel
from glrp.model.mnist import MnistModel
from spektral.data.loaders import MixedLoader

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from lib import graph, coarsening
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from spektral.layers import GCNConv, ChebConv



def main():
    # parse input arguments
    tf.__version__
    args = utils.get_args()
    try:
        # parse config
        conf = config.load_config(args.config)
        conf["path_to_feature_val"] = args.features
        conf["path_to_feature_graph"] = args.graph
        conf["path_to_labels"] = args.labels
        conf["precision"] = np.float32

        # TODO: check if config has minimum parameters assigned
        # TODO: comments/docs for the config file

    except:
        print("missing or invalid arguments")
        exit(0)

    config.show_config(conf)
    # -- create log/checkpoint directories
    
    # TODO: R preprocessing in here

    # get mnist data
    # data preparation
    # returns MnistDataset
    mnist = spektral.datasets.MNIST()
    #print(mnist.a.shape)
    mnist.a = ChebConv.preprocess(mnist.a)
    mnist.a = sp_matrix_to_sp_tensor(mnist.a)
    #print(mnist.a)
    #print(np.min(mnist.a.values))
    #print(mnist[0].x)
    #print(mnist[0].y)
    #print(mnist[0].x.shape)

    #print(mnist)
    #print(mnist.a)
    #print(mnist[0])
    #print(mnist[0].x)

    # build and train model
    model = MnistModel(conf, mnist)
    runner = MnistRunner(conf, model, mnist)
    runner.run()

    # TODO: get explanations and visualize results
    # explainer = MnistExplainer(conf, mnist)

    # explainer.visualize()
    # explainer.plot_things()

    
main()
