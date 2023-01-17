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

    #config.show_config(conf)

    # TODO: create log/checkpoint directories
    #dirnames = [conf["path_to_data"], conf["path_to_results"]]
    #storage.create_dirs(dirnames)

    np.__config__.show()

    # process input data
    data = DataModel(conf)
    data.init_data()

    # convert data to spektral input format
    #data.show_data_infos()
    #predef = ["GSM491222", "GSM615188", "GSM491210", "GSM615713", "GSM491237", "GSM441855", "GSM50105", "GSM177885"]
    predef = pd.read_csv("/home/lutzseba/Desktop/prm/projectmodul-ma-lutz/results/predicted_concordance.csv")
    print(predef)
    predef = list(predef['Patient ID'].values)
    print(predef)
    data.train_test_split(0.1, predefined_test_patients=predef)
    #np.save(conf["path_to_results"] + "test_data.npy", np.array(data.PI_test))

    model = GCNNModel(conf, data)
    #gcnn_runner = GCNNRunner(conf, model, data)
    #gcnn_runner.run()
    # model.build_model()
    #model.save("test_model.model")

    tf_model = model.load("test_model.model")
    #utils.save_concordance(conf, tf_model, data, "ge_concordance.csv")

    m = model.build_model()
    a = tf_model.get_weights()
    m.set_weights(a)
    #explainer = GCNNExplainer(conf, m, data)
    #relevances = explainer.explain()
    #np.save(conf["path_to_results"] + "test.npy", np.array(relevances))

    ## TODO: save/print relevance scores
    #results.create_relevance_score_csv(np.array(relevances), data, conf["path_to_results"] + "rel_score.csv")
    pi = np.load(conf["path_to_results"] + "test_data.npy")
    data.PI_test = pi
    rels = pd.read_csv(conf["path_to_results"] + "rel_score.csv", index_col=0)
    concordances = pd.read_csv(conf["path_to_results"] + "ge_concordance.csv")
    ndex = NdexManager(conf)
    net_cx, g = ndex.download_network("a420aaee-4be9-11ec-b3be-0ac135e8bacf")
    used_genes = [x[1].get('n') for x in net_cx.get_nodes()]
    results.generate_subnetwork_results(data, conf, rels, used_genes, concordances)

main()
