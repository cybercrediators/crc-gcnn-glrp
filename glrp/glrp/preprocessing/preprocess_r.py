import rpy2.robjects as robjects
import glob
import os
import pandas as pd
from glrp.preprocessing.rbridge import RBridge
from glrp.helpers import utils
from pathlib import Path


def preprocess_folder(f_path, output_fname):
    """
    Call the R preprocessing script and preprocess the data
    in the given file path.
    """
    rb = RBridge('./glrp/preprocessing/preprocess.R', 'preprocess_folder')
    rb.call(f_path, output_fname)

def preprocess_single(f_name, output_fname):
    """
    Call the R preprocessing script and preprocess the data
    from the given file.
    """
    rb = RBridge('./glrp/preprocessing/preprocess.R', 'preprocess_single')
    rb.call(f_name, output_fname)

def preprocess_scraped_data(f_path, output_fname):
    """
    Preprocess scraped data from the given folder with
    corresponding subfolders of sub-datasets.
    """
    # get all subfolders
    dirs = Path(f_path).iterdir()
    print("Get files...\nStart preprocessing...")
    for dir in dirs:
        # preprocesse each folder to a csv in the root dir
        dest = str(dir.absolute()) + "/" + str(dir.stem) + ".csv"
        if utils.check_file(dest):
            continue
        preprocess_folder(str(dir.resolve()) + "/", dest)
    # combine csv files in the root folder
    preprocessed_sets = glob.glob(f_path + "**/*.csv", recursive=True)
    sanitize_data(preprocessed_sets)
    print("Combining preprocessed datasets...")
    combined_set = combine_dataset(preprocessed_sets)
    print(combined_set)
    #combined_set.to_csv(f_path + "intermed_res.csv")
    print("Write combined set to file...")
    # apply quantile normalization to the combined dataset csv
    quantile_normalisation(combined_set, output_fname)


def combine_dataset(dataset_files: list):
    """
    Merge multiple preprocessed datasets into a single csv
    """
    # take first dataset and append others onto it
    if len(dataset_files) <= 0:
        print("No preprocessed files found!")
        return None
    ret = pd.read_csv(dataset_files[0], index_col=0)
    ret = ret.drop(columns=['Unnamed: 0'], errors='ignore')
    for dataset in dataset_files[1:]:
        add_set = pd.read_csv(dataset, index_col=0)
        add_set = add_set.drop(columns=['Unnamed: 0'], errors='ignore')
        ret = ret.merge(add_set, how='left')
    return ret

def quant_norm(df):
    ranks = (df.rank(method="first")
              .stack())
    rank_mean = (df.stack()
                   .groupby(ranks)
                   .mean())
    # Add interpolated values in between ranks
    finer_ranks = ((rank_mean.index+0.5).to_list() +
                    rank_mean.index.to_list())
    rank_mean = rank_mean.reindex(finer_ranks).sort_index().interpolate()
    return (df.rank(method='average')
              .stack()
              .map(rank_mean)
              .unstack())

def quantile_normalisation(combined_set, output_path):
    """
    Quantile normalize a given pandas dataframe
    """
    print(combined_set)
    col_names = [x for x in list(combined_set) if "CEL" in x]
    comb_set = combined_set.loc[:, col_names]
    print(comb_set)
    ret = quant_norm(comb_set)
    print(ret)
    combined_set[col_names] = ret[col_names]
    combined_set.to_csv(output_path)

def sanitize_data(datasets):
    """
    Remove unnotated rows and duplicates
    """
    for ds in datasets:
        df = pd.read_csv(ds)
        print(df)
        # drop N/A rows
        df = df.dropna()
        #print(df)
        df = df.groupby('geneID').max()
        df.set_index('Unnamed: 0')
        df.to_csv(ds) 

def preprocess_single(f_path, output_fname):
    """
    Call the R preprocessing script and preprocess the data
    in the given file path.
    """
    rb = RBridge('./preprocess.R', 'preprocess_single')
    rb.call(f_path, output_fname)


# folder_path = "/home/lutzseba/Desktop/ma/gcnn-and-grlp/glrp/glrp/preprocessing/test_folder/"
# output_path = "/home/lutzseba/Desktop/ma/gcnn-and-grlp/glrp/glrp/preprocessing/test_folder/test.csv"
# preprocess_scraped_data(folder_path, output_path)

#f_path = "/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/crc_extended/data/"
#lf_path = "/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/crc_extended/"
##
#preprocessed_sets = glob.glob(f_path + "**/*.csv", recursive=True)
#preprocessed_sets.remove("/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/crc_extended/data/GSE8671/GSE8671.csv")
#preprocessed_sets.insert(0, "/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/crc_extended/data/GSE8671/GSE8671.csv")
##print(preprocessed_sets)
##
##sanitize_data(preprocessed_sets)
##cs = combine_dataset(preprocessed_sets)
#cs = pd.read_csv(lf_path + "intermed_res.csv")
#cd = cs.loc[:,~cs.columns.duplicated()].copy()
#cs = cs.fillna(0.0)
#quantile_normalisation(cs, lf_path + "intermed_res.csv")
#cs.to_csv(lf_path + "intermed_res.csv")

#print(df)
#print(df)
#print(len(df.columns))
