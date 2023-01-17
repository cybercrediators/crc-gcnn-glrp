import sys
import re
import csv
import pandas as pd
import numpy as np


def write_output(fname, data):
    """write to mapped csv file"""
    df = pd.DataFrame.from_dict(data)
    df.to_csv(fname, index=False)

def map_preprocessed_to_gene_list(preprocessed, genes):
    """
    Map preprocessed gene expressions onto a list of genes and
    return the resulting dict.
    """
    new_col = {}
    for column in preprocessed:
        col = re.split('\_|\.', column)[0]
        new_col_vals = []
        # Skip dataset names
        if "GSM" not in column:
            continue
        print(col, " doing")
        for gene in genes:
            # check if preprocessed data has values for the currently selected gene
            idx = preprocessed.index[preprocessed['geneSymbols'] == gene]
            # check if multiple probes match to the gene and assign them to the resulting column
            # use the probe average value
            if len(idx) <= 0:
                new_col_vals.append(0.0)
                continue
            elif len(idx) > 1:
                asdf = np.array([preprocessed[[column]].loc[z] for z in idx])
                max_val = np.mean(asdf)
                new_col_vals.append(max_val)
            else:
                idx = idx[0]
                new_col_vals.append(preprocessed[[column]].loc[idx][0])
        new_col[col] = new_col_vals
    # add probe column at the end, should be in the right order anyways
    new_col["probes"] = genes
    return new_col

def map_preprocessed_to_graph(preprocessed_path, graph_path, output_file):
    """
    Map a given preprocessed file onto a given network graph and write
    the output to the given output file name.
    """
    # INPUT: preprocessed data, graph, [output file name]
    # open graph adjacency matrix
    with open(graph_path, newline='') as f:
        reader = csv.reader(f)
        genes = next(reader)

    # read preprocessed gene expressions
    preprocessed = pd.read_csv(preprocessed_path)

    # merge only active genes from the preprocessed data to a merge csv file
    # print("Merge active genes...")
    # preprocessed_genes = preprocessed['geneSymbols'].to_numpy()
    # preprocessed_genes = preprocessed_genes[preprocessed_genes == preprocessed_genes]
    # preprocessed_genes = np.unique(preprocessed_genes)
    # outp = map_preprocessed_to_gene_list(preprocessed, preprocessed_genes)
    # write_output(MERGE_OUTPUT, outp)

    print("Map genes on graph...")
    # delete unavailable genes found in adjacency matrix, but not in preprocessed
    preprocessed = preprocessed[preprocessed['geneSymbols'].isin(genes)]
    outp = map_preprocessed_to_gene_list(preprocessed, genes)
    write_output(output_file, outp)
#lf_path = "/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/crc_extended/"
#map_preprocessed_to_graph(lf_path + "intermed_res.csv", lf_path + "HPRD_PPI.csv", lf_path + "crc_ext_mapped.csv")
