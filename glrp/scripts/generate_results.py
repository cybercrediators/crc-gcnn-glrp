import os
import re
import sys
import pandas as pd
import json
import numpy as np
from collections import Counter, defaultdict
import seaborn as sns
from pylab import savefig
import imgkit
import networkx as nx
import ndex2

if len(sys.argv) != 6:
    print("Please provide predicted classes, relevance scores, subtype predictions and mapped data and ndex network ID!")
    sys.exit(1)
PREDICTED = sys.argv[1]
RELEVANCES = sys.argv[2]
SUBTYPES = sys.argv[3]
MAPPED = sys.argv[4]
GRAPH_ID = sys.argv[5]
N = 100


# create directories if they don't exist
def create_dir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# write json data to a file
def write_json(name, data):
    with open(name, 'w') as fp:
        json.dump(data, fp)


# generate a patient dicitonary for output
def generate_patient_dict(patients, subtypes, label):
    res = []
    for patient in patients:
        p = {}
        p['name'] = patient
        sub_tmp = subtypes[subtypes['Unnamed: 0'].str.match(patient)]['prediction'].values[0]
        p['mfsYears'] = label
        p['subtype'] = sub_tmp if sub_tmp in ['CMS1', 'CMS2', 'CMS3', 'CMS4'] else 'none'
        # here just label 0/1 if cancer or not instead of years
        res.append(p)
    return res


# get the most N relevant selected genes for the given patients and relevance scores
def get_gene_occurences(relevances, correct_patients, N):
    relevances = relevances.loc[relevances['Patient ID'].isin(correct_patients)]
    relevances_np = relevances.to_numpy()[:, 1:]
    indices = relevances_np.argsort()[:, :-N-1:-1]
    
    selected_genes = []
    idx2 = 0
    for idx, patient in relevances.iterrows():
        r = patient.iloc[indices[idx2]+1]
        selected_genes.extend(r.index.values)
        idx2 += 1
    return dict(Counter(selected_genes))


patients = {}

# get correctly classified patient ids
predicted = pd.read_csv(PREDICTED)
# predicted = predicted.loc[predicted['Concordance'] == 1]

correct_patients = predicted['Patient ID'].values
cancer_patients = predicted.loc[predicted['label'] == 1]['Patient ID'].values
normal_patients = predicted.loc[predicted['label'] == 0]['Patient ID'].values

# select relevnances for correctly classified patients
relevances = pd.read_csv(RELEVANCES)

# get most selected genes from the relevance scores for the predicted patients
occurences = {}
occurences["Cancer"] = get_gene_occurences(relevances, cancer_patients, 20)
occurences["Non-Cancer"] = get_gene_occurences(relevances, normal_patients, 20)
occ = pd.DataFrame(occurences)
occ.to_csv("top_occurences.csv")

# get patients by subtypes
subtypes = pd.read_csv(SUBTYPES)
# select only the given patients and strip gene names
subtype_predictions = subtypes[subtypes['Unnamed: 0'].str.match('|'.join(correct_patients))][['prediction', 'Unnamed: 0']].values
subtype_predictions[:, 1] = [re.split('\_|\.', x)[0] for x in subtype_predictions[:, 1]]
subtype_predictions = dict(subtype_predictions[:, ::-1])

mapped = pd.read_csv(MAPPED)

# expression levels by quantile normalization using pandas quantile
probes = mapped['probes'].values
# transpose the mapped file
mapped = mapped[correct_patients].T
# calculate the quantile values for 25% and 75%
quantile_levels = mapped.quantile([0.25, 0.75])

# replace quantile values by categories
for key, val in quantile_levels.items():
    asdf = mapped[key].values
    asdf = ['LOW' if x <= val[0.25] else 'NORMAL' if x > val[0.25] and x <= val[0.75] else 'HIGH' for x in asdf]
    mapped[key] = asdf

cols = {v: k for v, k in enumerate(probes)}
mapped.columns = probes

# add a subtype and label column to the expression level csv
sorted_subtypes = [subtype_predictions[x] for x in correct_patients]
mapped.insert(0, 'subtype', sorted_subtypes)
mapped.insert(1, 'class', predicted['Predicted'].values)

mapped.to_csv('expression_levels.csv')
expression_levels = mapped

# calculate metarelsubnet data, based on the 100 genes with the highest relevance

# create directories
create_dir('./data')
create_dir('./data/patient')

mapped = pd.read_csv(MAPPED)
relevances = pd.read_csv(RELEVANCES)
mapped = mapped.iloc[:, :-1]
# create patients.json
patients = {}
# add cancer/non-cancer patients (named metastatic/nonmetastatic bc of the metarelsubnetvisualizer)
patients["metastatic"] = generate_patient_dict(cancer_patients, subtypes, 1)
patients["nonmetastatic"] = generate_patient_dict(normal_patients, subtypes, 0)

# extract the minimum/maximum expression level
patients["geMin"] = min(mapped[mapped > 0.0].min().values)
patients["geMax"] = max(mapped.max().values)

write_json('data/patients.json', patients)

# extract minimum and maximum relevance scores and create thresholds.json
thresholds = {}
tmp = relevances.iloc[:, 1:]
max_rel = max(tmp.max().values)
min_rel = min(tmp[tmp > 0.0].min().values)

thresh = {"threshold": min_rel, "max": max_rel}
# metastatic/nonmetastatic because of the metarelsubnetvisualizer
thresholds["metastatic"] = thresh
thresholds["nonmetastatic"] = thresh
write_json('data/thresholds.json', thresholds)

# get most relevant calculated genes to use in the visualisation based on the N most relevant genes for each patient
most_relevant = get_gene_occurences(relevances, correct_patients, N)
most_relevant_genes = list(most_relevant.keys())

# create network.json cytoscape network
network = {}
nodes = []
grouped_subtypes = {}

# group subtype prediction occurences by each subtype
subtype_predictions = {k: 'none' if v not in ['CMS1', 'CMS2', 'CMS3', 'CMS4'] else v for k, v in subtype_predictions.items() }
for key, value in subtype_predictions.items():
    grouped_subtypes.setdefault(value, set()).add(key)
occ_na = get_gene_occurences(relevances, list(grouped_subtypes['none']), N)
occ_CMS1 = get_gene_occurences(relevances, list(grouped_subtypes['CMS1']), N)
occ_CMS2 = get_gene_occurences(relevances, list(grouped_subtypes['CMS2']), N)
occ_CMS3 = get_gene_occurences(relevances, list(grouped_subtypes['CMS3']), N)
occ_CMS4 = get_gene_occurences(relevances, list(grouped_subtypes['CMS4']), N)
occ_non_cancer = get_gene_occurences(relevances, normal_patients, N)
occ_all = get_gene_occurences(relevances, correct_patients, N)

# add nodes/patients to the network
for gene in most_relevant_genes:
    node = {}
    occ = {}
    occ["all"] = occ_all[gene] if gene in occ_all else 0
    occ["none"] = occ_na[gene] if gene in occ_na else 0
    occ["CMS1"] = occ_CMS1[gene] if gene in occ_CMS1 else 0
    occ["CMS2"] = occ_CMS2[gene] if gene in occ_CMS2 else 0
    occ["CMS3"] = occ_CMS3[gene] if gene in occ_CMS3 else 0
    occ["CMS4"] = occ_CMS4[gene] if gene in occ_CMS4 else 0
    occ["Normal"] = occ_non_cancer[gene] if gene in occ_non_cancer else 0
    node["data"] = {"id": gene}
    node["occ"] = occ
    nodes.append(node)


# gather all edges for every gene from the network, add source and target genes
edges = []
client = ndex2.client.Ndex2()
client_resp = client.get_network_as_cx_stream(GRAPH_ID)
net_cx = ndex2.create_nice_cx_from_raw_cx(json.loads(client_resp.content))
node_list = list(net_cx.get_nodes())
idx = 0

# loop through the edge list and add all edges with source/target from
# the most relevant genes
for edge_id, edge in net_cx.get_edges():
    e = {}
    # get target and source of each edge
    target = node_list[edge.get('t')][1].get('n')
    source = node_list[edge.get('s')][1].get('n')
    if target in most_relevant_genes and source in most_relevant_genes:
        idx += 1
        e["data"] = {"id": idx, "source": source, "target": target}
        edges.append(e)

network["nodes"] = nodes
network["edges"] = edges
write_json('data/network.json', network)

# create patients/[patient].json files
for patient in correct_patients:
    p = []
    # get N most relevant genes for this patient
    rel_scores = relevances[relevances["Patient ID"] == patient].to_numpy()[0][1:]
    # get indices to get data from the corresponding relevance map
    rel_indices = np.argpartition(rel_scores, -N)[-N:][::-1]
    genes = relevances.columns[1:][rel_indices]
    ge_scores = mapped[patient][rel_indices].values
    for idx, gene in enumerate(genes):
        g = {}
        # build each patient dictionary
        score = relevances[relevances["Patient ID"] == patient][gene].values[0]
        g["name"] = gene
        g["score"] = score
        g["ge"] = ge_scores[idx]
        g["geLevel"] = expression_levels[gene][patient]
        g["mtb"] = False
        p.append(g)
    write_json('data/patient/'+patient+'.json', p)
