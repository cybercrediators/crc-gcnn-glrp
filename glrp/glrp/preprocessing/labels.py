import re
import csv
import pandas as pd
import copy


def generate_labels_from_preprocessed(preprocessed_fname, labels_fname, outp_fname):
    """
    Generate a label file from the preprocessed gene expressions and the given
    label list
    """

    # get csv labels from preprocessed data (col. headers)
    #with open(preprocessed_fname, newline='') as f:
    #    reader = csv.reader(f)
    #    used_samples = next(reader)
    #used_samples = used_samples[3:]
    # read assigned label samples
    with open(labels_fname) as f:
        lines = f.read().splitlines()
    samples = dict(label.split(" ") for label in lines)
    used_samples = list(samples.keys())

    df = pd.read_csv(preprocessed_fname)
    df = df.rename(columns=lambda x: re.split('\_|\.', x)[0])
    retain_samples = copy.copy(used_samples)
    retain_samples.insert(0, 'geneSymbols')

    # strip to basename
    df = df[df.columns.intersection(retain_samples)]
    df.to_csv(preprocessed_fname)
    #used_samples = [re.split('\_|\.', k)[0] for k in used_samples]
    # print(used_samples)

    # assign found labels from samples
    # add them if they exist in the preprocessed data
    vals = [samples[k] for k in used_samples]
    res = dict(zip(used_samples, vals))

    # write back csv
    with open(outp_fname, 'w') as f:
        w = csv.DictWriter(f, res.keys())
        w.writeheader()
        w.writerow(res)
#lf_path = "/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/crc_extended/"
#generate_labels_from_preprocessed(lf_path + "intermed_res.csv", lf_path + "labels.txt", lf_path + "crc_ext_labels.csv")
