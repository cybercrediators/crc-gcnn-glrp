import sys
import re
import csv
import pandas as pd
import numpy as np
# INPUT: preprocessed data, graph, [output file name]
PREPROCESSED = sys.argv[1]
GRAPH = sys.argv[2]
if len(sys.argv) < 4:
    OUTPUT = "test_mapped_2.csv"
else:
    OUTPUT = sys.argv[3]

if len(sys.argv) < 5:
    MERGE_OUTPUT = "test_merged.csv"
else:
    MERGE_OUTPUT = sys.argv[4]

# open graph adjacency matrix
with open(GRAPH, newline='') as f:
    reader = csv.reader(f)
    genes = next(reader)
#print(genes)
# read preprocessed gene expressions
preprocessed = pd.read_csv(PREPROCESSED)

def map_preprocessed_to_gene_list(preprocessed, genes):
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

def write_output(fname, data):
    # write to mapped csv file
    df = pd.DataFrame.from_dict(data)
    df.to_csv(fname, index=False)


# merge only active genes from the preprocessed data to a merge csv file
#print("Merge active genes...")
#preprocessed_genes = preprocessed['geneSymbols'].to_numpy()
#preprocessed_genes = preprocessed_genes[preprocessed_genes == preprocessed_genes]
#preprocessed_genes = np.unique(preprocessed_genes)
#outp = map_preprocessed_to_gene_list(preprocessed, preprocessed_genes)
#write_output(MERGE_OUTPUT, outp)

print("Map genes on graph...")
# delete unavailable genes found in adjacency matrix, but not in preprocessed
preprocessed = preprocessed[preprocessed['geneSymbols'].isin(genes)]
outp = map_preprocessed_to_gene_list(preprocessed, genes)
write_output(OUTPUT, outp)
