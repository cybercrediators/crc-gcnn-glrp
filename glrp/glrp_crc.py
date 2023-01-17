import tensorflow as tf
import numpy as np
import pandas as pd
import pprint
from glrp.helpers import config, utils, storage
from glrp.model.gcnn import GCNNModel
from glrp.model.testing import TestingModel
from glrp.trainer.gcnn_runner import GCNNRunner
from glrp.trainer.test_runner import TestingRunner
from glrp.data.data_model import DataModel
from glrp.postprocessing.gcnn_explainer import GCNNExplainer
from glrp.postprocessing import results
from glrp.data.ndex import NdexManager

from lib import coarsening


def main():
    # parse input arguments
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

    # np.__config__.show()

    # process input data
    data = DataModel(conf)
    data.init_data()

    # convert data to spektral input format
    #data.show_data_infos()
    #predef = ["GSM491222", "GSM615188", "GSM491210", "GSM615713", "GSM491237", "GSM441855", "GSM50105", "GSM177885"]
    data.train_test_split(0.1, seed=42)
    #np.save(conf["path_to_results"] + "test_data.npy", np.array(data.PI_test))

    model = GCNNModel(conf, data)
    #gcnn_runner = GCNNRunner(conf, model, data)
    #gcnn_runner.run()
    # model.build_model()
    #model.save("crc_model.model")

    tf_model = model.load("crc_model.model")
    #utils.save_concordance(conf, tf_model, data, "crc_concordance.csv")

    m = model.build_model()
    a = tf_model.get_weights()
    m.set_weights(a)
    explainer = GCNNExplainer(conf, m, data)
    relevances = explainer.explain()
    #np.save(conf["path_to_results"] + "test.npy", np.array(relevances))

    ## TODO: save/print relevance scores
    #results.create_relevance_score_csv(np.array(relevances), data, conf["path_to_results"] + "rel_score.csv")
    #subtypes = pd.read_csv("/home/lutzseba/Desktop/prm/projectmodul-ma-lutz/colectoral_data/dataset/subtype_pred.csv", index_col=0)
    #pi = np.load(conf["path_to_results"] + "test_data.npy")
    #data.PI_test = pi
    #rels = pd.read_csv(conf["path_to_results"] + "rel_score.csv", index_col=0)
    #concordances = pd.read_csv(conf["path_to_results"] + "crc_concordance.csv")
    #patients = concordances['Patient ID'].values
    #used_genes, _ = results.get_gene_occurences(rels, patients, 300)
    #results.generate_subnetwork_results(data, conf, rels, used_genes, concordances, subtypes)


main()
