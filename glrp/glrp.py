import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pprint
import time
from glrp.helpers import config, utils, storage
from glrp.model.gcnn import GCNNModel
from glrp.model.testing import TestingModel
from glrp.trainer.gcnn_runner import GCNNRunner
from glrp.trainer.test_runner import TestingRunner
from glrp.data.data_model import DataModel
from glrp.postprocessing.gcnn_explainer import GCNNExplainer
from glrp.postprocessing import results
from glrp.preprocessing import preprocess_r
from glrp.data.ndex import NdexManager

from lib import coarsening
from pathlib import Path

def main():
    """
    Main glrp tool loop.
    CLI options -> helper/utils.py
    """
    full_start_time = time.time()

    # parse input arguments
    args = utils.get_args()
    #print(args)
    if args.config == 'None':
        print("No config file was provided. Please provide a config file!")
        exit(1)

    # read config
    try:
        # parse config
        conf = config.load_config(args.config)
        # TODO: remove legacy
        conf["precision"] = np.float32
        # create data folders
        to_create = []
        print("Creating non-existing directories...")
        for key in conf.keys():
            if "path" in key or "base_data" in key:
                to_create.append(conf[key])
        storage.create_dirs(to_create)
    except:
        print("missing or invalid arguments")
        exit(1)

    print("TIME after init: {}".format(time.time() - full_start_time))

    start_time = time.time()
    # perform preprocessing
    if args.preprocessing or args.full:
        print("Perform preprocessing...")
        conf = utils.preprocessing_command(conf, args)
        config.save_config(args.config, conf)
        print("TIME after preprocessing: {}, complete run time: {}".format(time.time() - start_time, time.time() - full_start_time))
    else:
        print("Skipping preprocessing...")

    start_time = time.time()
    conf["path_to_feature_val"] = conf["path_to_data"] + conf["mapped_csv_name"]
    conf["path_to_feature_graph"] = conf["path_to_data"] + conf["network_csv_name"]
    conf["path_to_labels"] = conf["path_to_data"] + conf["label_csv_name"]

    conf["model_path"] = conf["path_to_results"] + conf["model_name"]
    conf["concordance_path"] = conf["path_to_results"] + conf["concordance_csv_name"]

    # process input data
    data = DataModel(conf)
    # convert data to spektral input format
    data.init_data()

    # check for predefined patients
    seed = None
    predef = None
    #data.show_data_infos()
    #predef = ["GSM491222", "GSM615188", "GSM491210", "GSM615713", "GSM491237", "GSM441855", "GSM50105", "GSM177885"]
    #predef = pd.read_csv("/home/lutzseba/Desktop/prm/projectmodul-ma-lutz/results/predicted_concordance.csv")
    #print(predef)
    #predef = list(predef['Patient ID'].values)
    #print(predef)

    if args.predef_patients != 'None':
        if utils.check_file(args.predef_patients):
            predef = utils.read_txt(args.predef_patients)
        else:
            print("Predefined patients not found!")
            exit(1)
    if args.test_seed != 'None':
        seed = int(args.test_seed)

    print(seed, predef)
        
    # train/test split
    data.train_test_split(conf["test_split"], predefined_test_patients=predef, seed=seed)
    # save pi backup
    np.save(conf["path_to_results"] + "pi_data.npy", np.array(data.PI_test))
    # create model
    model = GCNNModel(conf, data)
    print("TIME after data model init: {}, complete run time: {}".format(time.time() - start_time, time.time() - full_start_time))
    m = model.build_model()
    m.summary()

    if args.predict:
        if len(args.predict) > 2 or len(args.predict) < 1:
            print("Wrong prediction params! [input CEL filepath] [(opt.) alternate model path]")
            exit(1)
        if len(args.predict) == 2:
            conf["model_path"] = args.predict[1]
        print("Preprocessing and predicting input data on the given model ({})".format(conf["model_path"]))
        f_path = args.predict[0]
        f_folder = os.path.dirname(f_path) + "/"
        f_preproc = f_folder + Path(f_path).stem + ".csv"
        print("Preprocess single patient ({})".format(f_path, f_preproc))
        preprocess_r.preprocess_single(f_path, f_preproc)
        tf_model = model.load(conf["model_path"])
        m = model.build_model()
        a = tf_model.get_weights()
        m.set_weights(a)
        exit(0)

    if args.get_stats:
        print("+++ STATS +++")
        utils.get_stats(conf, data, model.build_model())
        return

    start_time = time.time()
    # train model on preprocessed data
    if args.train or args.full:
        gcnn_runner = GCNNRunner(conf, model, data)
        gcnn_runner.run()
        #m = model.build_model()
        #m.summary()
        #model.save(conf["model_path"])
        utils.save_concordance(conf, model.model, data, conf["concordance_path"])
        print("TIME after training: {}, complete run time: {}".format(time.time() - start_time, time.time() - full_start_time))

    if args.save_concordance or args.full or args.visualize:
        tf_model = model.load(conf["model_path"])
        m = model.build_model()
        a = tf_model.get_weights()
        m.set_weights(a)
        utils.save_concordance(conf, m, data, conf["concordance_path"])

    start_time = time.time()
    if args.explain or args.full:
        tf_model = model.load(conf["model_path"])
        m = model.build_model()
        a = tf_model.get_weights()
        m.set_weights(a)
        explainer = GCNNExplainer(conf, m, data)
        relevances = explainer.explain()
        # backup
        np.save(conf["path_to_results"] + "rel_data.npy", np.array(relevances))
        # create csv
        results.create_relevance_score_csv(np.array(relevances), data, conf["path_to_results"] + conf["relevance_csv_name"])
        print("TIME after relevance propagation: {}, complete run time: {}".format(time.time() - start_time, time.time() - full_start_time))

    start_time = time.time()
    if args.visualize or args.full:
        #pi = np.load(conf["path_to_results"] + "pi_data.npy")
        #data.PI_test = pi
        rels = pd.read_csv(conf["path_to_results"] + conf["relevance_csv_name"], index_col=0)
        concordances = pd.read_csv(conf["concordance_path"])
        #ndex = NdexManager(conf)
        #net_cx, g = ndex.download_network("a420aaee-4be9-11ec-b3be-0ac135e8bacf")
        #used_genes = [x[1].get('n') for x in net_cx.get_nodes()]
        patients = concordances['Patient ID'].values
        if args.used_genes:
            used_genes = utils.read_txt(args.used_genes)
        else:
            used_genes, _ = results.get_gene_occurences(rels, patients, 200)
        subtypes = pd.read_csv(conf["path_to_data"] + conf["subtype_csv_name"], index_col=0)
        results.generate_subnetwork_results(data, conf, rels, used_genes, concordances, subtypes=subtypes)
        print("TIME after result generation: {}, complete run time: {}".format(time.time() - start_time, time.time() - full_start_time))

main()
