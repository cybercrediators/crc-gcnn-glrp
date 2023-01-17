import argparse
import click
import pandas as pd
import numpy as np
from spektral.data.loaders import MixedLoader
from spektral.utils.sparse import sp_matrix_to_sp_tensor
from lib import coarsening
from pathlib import Path

from glrp.preprocessing import scrape, preprocess_r, subclass_prediction, mapping, labels
from glrp.data.ndex import NdexManager
from glrp.helpers.config import save_config


def get_args():
    """
    Retrieve and return the given command line arguments.
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-p', '--preprocessing',
        action='store_true',
        help='Perform the necessary preprocessing steps on the data folder from the config file.',
    )
    argparser.add_argument(
        '--skip-scraper',
        action='store_true',
        help='Skip the scraping part if you already downloaded and unpacked the gene expression data.',
    )
    argparser.add_argument(
        '--skip-r-preprocessing',
        action='store_true',
        help='Skip the Preprocessing steps in R.',
    )
    argparser.add_argument(
        '-f', '--full',
        action='store_true',
        help='Full execution from preprocessing to training',
    )
    argparser.add_argument(
        '-t', '--train',
        action='store_true',
        help='Train a model from the given data in the config file',
    )
    argparser.add_argument(
            '--predef-patients',
            default='None',
            help='Create a test set from a predefined list of patients'
    )
    argparser.add_argument(
            '--test-seed',
            default='None',
            help='Create a test set from a predefined test seed'
    )
    argparser.add_argument(
            '--save-concordance',
            action='store_true',
            help='Save the concordance from the given model and data in the config file'
    )
    argparser.add_argument(
        '-o', '--predict',
        nargs='+',
        default=[],
        help='--predict [preprocessed patient] [model name] -- predict the given patient data on an existing model',
    )
    argparser.add_argument(
        '--used-genes',
        default=None,
        help='--used-genes [path-to-gene-list] -- genes to use in the visualisation',
    )
    argparser.add_argument(
            '--get-stats',
            action='store_true',
            help="Get stats about data, model etc."
    )
    argparser.add_argument(
            '-e', '--explain',
            action='store_true',
            help="Explain the given data and the config file. For single data prediction/explanation use the predict option."
    )
    argparser.add_argument(
            '-v', '--visualize',
            action='store_true',
            help="Generate results and upload the generated results to ndex."
    )
    argparser.add_argument(
            '-c', '--config',
            metavar='C',
            default='None',
            help='Define config file')
    args = argparser.parse_args()
    return args

def check_config_keys(config, keys):
    """
    Check if config contains all given keys
    """
    pass

def preprocessing_command(config, args):
    """
    Perform the preprocessing command -p from the cli
    """
    # download codes/code
    if not args.skip_scraper:
        c_name = config["path_to_data"] + "codes.txt"
        if not check_file(c_name):
            print("codes.txt not found!")
            c_name = click.prompt("Please provide a txt file: ", type=str)
            if not check_file(c_name):
                print("File not found!")
                exit(1)
        print("Get codes from codes file: ({})".format(c_name))
        sc = scrape.GeoScraper(config)
        # unpack files
        sc.full_prepare_data(c_name)

    if not args.skip_r_preprocessing:
        # r preprocessing
        if "preprocessed_csv_name" not in config.keys():
            config["preprocessed_csv_name"] = "preprocessed.csv"
        preprocessed_path = config["path_to_data"] + config["preprocessed_csv_name"]
        print("Start preprocessing, saving to: {}".format(preprocessed_path))
        if not check_file(preprocessed_path):
            preprocess_r.preprocess_scraped_data(config["base_data_folder"], preprocessed_path)
        print("Preprocessing done, saved to: {}".format(preprocessed_path))

        # r subtype prediction
        print("Start subtype prediction...")
        if "subtype_csv_name" not in config.keys():
            config["subtype_csv_name"] = "subtypes.csv"
        subtype_path = config["path_to_data"] + config["subtype_csv_name"]
        if not check_file(subtype_path):
            subclass_prediction.subtype_prediction(config["base_data_folder"], subtype_path)
        print("Subtype prediction done, saved to: {}".format(subtype_path))

    preprocessed_path = config["path_to_data"] + config["preprocessed_csv_name"]
    # download mapping graph
    if "network_csv_name" not in config.keys():
        config["network_csv_name"] = "network.csv"
    network_path = config["path_to_data"] + config["network_csv_name"]
    ndex_manager = NdexManager(config)
    if not check_file(network_path):
        ndex_manager.save_adj(network_path, config["net_cx_sourceNetwork"])
    print("Network downloaded, saved to: {}".format(network_path))

    # mapping
    if "mapped_csv_name" not in config.keys():
        config["mapped_csv_name"] = "mapped.csv"
    mapped_path = config["path_to_data"] + config["mapped_csv_name"]
    if not check_file(mapped_path):
        mapping.map_preprocessed_to_graph(preprocessed_path, network_path, mapped_path)
    print("Mapping done, saved to: {}".format(network_path))

    # labels (must be provided)
    txt_label_path = config["path_to_data"] + config["label_txt_name"]
    if "label_csv_name" not in config.keys():
        config["label_csv_name"]
        
    if not check_file(txt_label_path):
        print("No labels txt file found! Please create them now...")
        click.confirm('Did you create the labels?', abort=True)
    csv_label_path = config["path_to_data"] + config["label_csv_name"]
    if not check_file(csv_label_path):
        labels.generate_labels_from_preprocessed(preprocessed_path, txt_label_path, csv_label_path)
    print("Labeling done, saved to: {}".format(csv_label_path))

    # write the output file names to the cli and config file
    return config

def check_file(dir_name):
    """Check if the given directory/file exists"""
    f = Path(dir_name)
    if f.exists():
        return True
    return False

def save_concordance(config, model, data, output_file):
    """
    Create a concordance csv file from the given model predictions
    and the current data.
    """
    # data loader
    loader = MixedLoader(data.test_data, batch_size=data.test_data.n_graphs, shuffle=False, epochs=1)
    concordance = {"Patient ID": [], "label": [], "pred": [], "concordance": []}
    print(loader)

    ## evaluate data
    for i, batch in enumerate(loader):
        inputs, target = batch
        print(target)
        inputs = (inputs[0], sp_matrix_to_sp_tensor(data.adj_coarsened[0]), sp_matrix_to_sp_tensor(data.adj_coarsened[1]))
        pred = model(inputs, training=False)
        pred_classes = np.array(np.argmax(pred, axis=1), dtype=np.float32)
        concordance["pred"].extend(pred_classes)
        concordance["label"].extend(target)
        concordance["concordance"].extend(np.invert(np.logical_xor(pred_classes, target)).astype(np.float32))
        ids = data.feature_values.iloc[:0, data.PI_test].columns.to_list()
        concordance["Patient ID"].extend(ids)

    #print(concordance)


    conc = pd.DataFrame.from_dict(concordance)
    conc.to_csv(output_file, index = False, header=True)

def get_stats(conf=None, data=None, model=None):
    """
    Print various stats about, data, model etc.
    """
    if data:
        print("+++ DATA STATS +++")
        print("Train patients: ", data.train_data)
        print("Test patients: ", data.test_data)
    if model:
        print("+++ MODEL SUMMARY +++")
        model.summary()

def read_txt(fname):
    """Read txt as list"""
    with open(fname, 'r') as f:
        ret = f.read().splitlines()
    return ret
