import numpy as np
import pandas as pd
import copy
import math
import re
from glrp.data.ndex import NdexManager
from collections import Counter, defaultdict

def create_relevance_score_csv(relevances, data, output_path="relevance_scores.csv"):
    """
    Creates a csv file with the relevances scores
    mapped to the given patient ids and ppi genes
    """
    # patient names
    rows = data.feature_values.iloc[:0, data.PI_test].columns.to_list()
    # gene names as rows
    genes = data.feature_graph.columns.to_list()
    relevances = relevances.reshape((relevances.shape[0], relevances.shape[1]))
    out = pd.DataFrame(relevances, columns=genes, index=rows)
    out.to_csv(output_path)

def get_patient_names(data):
    """
    Return all sample names from the given data object.
    """
    return data.feature_values.iloc[:0, data.PI_test].columns.to_list()

# get the most N relevant selected genes for the given patients and relevance scores
def get_gene_occurences(relevances, correct_patients, N):
    """
    Retrieve the N most selected genes from the given
    relevance scores and a list of patients.
    Returns:
        - dictionary: {Gene: #occ_num}
        - dictionary: {Patient: [N_most_relevant_genes]}
    """
    #relevances = relevances.loc[relevances['Patient ID'].isin(correct_patients)]
    # print(relevances)
    relevances = relevances.loc[correct_patients]
    relevances_np = relevances.to_numpy()[:, 1:]
    indices = relevances_np.argsort()[:, :-N-1:-1]
    
    selected_genes = []
    patient_occs = {}
    idx2 = 0
    for idx, patient in relevances.iterrows():
        r = patient.iloc[indices[idx2]+1]
        patient_occs[patient.name] = list(r.index.values)
        selected_genes.extend(r.index.values)
        idx2 += 1
    return dict(Counter(selected_genes)), patient_occs

def get_subtype_information(patients, subtypes):
    """
    Get the patients grouped by subtypes.
    """
    subtype_list = []
    pvalues = {}
    assigned = {}
    #print("\n\n\n\n\n")
    #print(patients)
    for idx, patient in subtypes.iterrows():
        name = re.split('\_|\.', idx)[0]
        if name in patients:
            z = np.where(patients == name)
            #print(z[0][0])
            #print(idx, patient)
            #print(name)
            subtype = "Normal" if patient["prediction"] not in ["CMS1", "CMS2", "CMS3", "CMS4"] else patient["prediction"]
            # subtype_list.insert(z[0][0], subtype)
            assigned[name] = subtype
            pvalues[name] = patient["p.value"]
    for p in patients:
        if p in assigned.keys():
            subtype_list.append(assigned[p])
        else:
            subtype_list.append("Normal")
    # print(patients)
    # print(subtype_list)
    return subtype_list, pvalues

def get_occurence_by_subtype(relevances, subtype_list, patient_list, N=20):
    """
    Retrieve occurences by top patient occurences
    Warning: subtype_list and patient_list must have the correct alignment
    """
    occs = {}
    patients = group_patients_by_subtype(subtype_list, patient_list)
    for key, value in patients.items():
        occs[key], _ = get_gene_occurences(relevances, value, N)
    return occs

def get_occurence_by_label(relevance_scores, cancer_patients, non_cancer_patients, N=20):
    """
    Retrieve top occurences by patient labels
    """
    occ = {}
    occ["Cancer"], _ = get_gene_occurences(relevance_scores, list(cancer_patients), N)
    occ["Non-Cancer"], _ = get_gene_occurences(relevance_scores, list(non_cancer_patients), N)
    return occ

def group_patients_by_subtype(subtype_list, patient_list):
    """
    Return the patients grouped by subtypes from the
    given subtype and patient list (must be aligned).
    """
    patients = defaultdict(list)
    for key, val in zip(subtype_list, patient_list):
        patients[key].append(val)
    return dict(patients)

def get_patient_subtype(patient_name, subtypes):
    """
    Get a subtype for a given patient name.
    """
    for idx, patient in subtypes.iterrows():
        name = re.split('\_|\.', idx)[0]
        if name == patient_name:
            return "Normal" if patient["prediction"] not in ["CMS1", "CMS2", "CMS3", "CMS4"] else patient["prediction"] 

def count_subtype_occurence(subtypes, patient_occs, patient_list, subtype_list, gene_list):
    """
    Return the given occurences grouped by the patients subtypes.
    """
    occs = {}
    #st = {"CMS1": 0, "CMS2": 1, "CMS3": 2, "CMS4": 3, "Normal": 4}
    #st = {"Basal": 0, "Her2": 1, "LumA": 2, "LumB": 3, "Normal": 4}
    patient_groups = group_patients_by_subtype(subtype_list, patient_list)
    for gene in gene_list:
        counts = [0, 0, 0, 0, 0]
        for type, names in patient_groups.items():
            for name in names:
                if gene in patient_occs[name]:
                    counts[subtypes[type]] += 1
        occs[gene] = counts
    return occs

def generate_occurences(relevance_scores, concordance, subtypes, config):
    """
    Generate gene occurences by patient subtypes.
    """
    conc = concordance[concordance['concordance'] != 0.0]
    correct_patients = conc['Patient ID'].values
    cancer_patients = conc[conc["pred"] == 1.0]['Patient ID'].values
    non_cancer_patients = conc[conc["pred"] == 0.0]['Patient ID'].values

    # save gene occurences for each label and subtype individually
    label_occurences = get_occurence_by_label(relevance_scores, cancer_patients, non_cancer_patients, 20)
    occ_df = pd.DataFrame(label_occurences)
    occ_df.to_csv(config["path_to_results"] + "occs.csv")

    if subtypes is None:
        return
    subtype_list, _ = get_subtype_information(correct_patients, subtypes)

    subtype_occurences = get_occurence_by_subtype(relevance_scores, subtype_list, correct_patients, 20)
    occ_df = pd.DataFrame(subtype_occurences)
    occ_df.to_csv(config["path_to_results"] + "subtype_occs.csv")

def generate_subnetwork_results(data, config, relevance_scores, used_genes, concordance, subtypes=None, output_path=None):
    """
    Generate ndex and metarelsubnetvis compatible results for the given network and data.
    """
    ndex = NdexManager(config)
    # TODO: add other result generators e.g. most important genes
    # get network attributes such as patient names, subclass etc
    network_attributes = []
    keys = [x for x in list(config.keys()) if "net_cx" in x]
    for k in keys:
        network_attributes.append((k[7:], config[k]))

    #print(concordance)
    conc = concordance[concordance['concordance'] != 0.0]
    correct_patients = conc['Patient ID'].values
    correct_patients_labels = conc["pred"].values
    #print(conc)
    # get correctly predicted patients
    network_attributes.append(("Patients", list(correct_patients)))
    patient_groups = []
    for label in correct_patients_labels:
        if label == 1.0:
            patient_groups.append("Cancer")
        else:
            patient_groups.append("No-Cancer")
    network_attributes.append(("PatientGroups", patient_groups))

    # get patient class
    # TODO: Get subtypes from the config file
    print(subtypes)
    print(correct_patients)
    subtype_list, pvalues = get_subtype_information(correct_patients, subtypes)
    print(subtype_list)

    # subtype_list = [np.random.choice(["Basal", "Her2", "LumA", "LumB", "Normal"]) for i in range(len(correct_patients))]
    network_attributes.append(("PatientSubtype", subtype_list))
    # TODO: survival years necessary??
    network_attributes.append(("PatientSurvivalYears", list(correct_patients_labels)))
    
    # generate gene scores, quantiles
    patient_ge = data.feature_values[correct_patients].T
    #patient_score = relevance_scores
    patient_ge.columns = data.get_all_gene_names()
    quantile_levels = patient_ge.quantile([0.25, 0.5, 0.75])
    stdev = patient_ge.std()
    gene_mean = patient_ge.mean()
    ge_levels = copy.deepcopy(patient_ge)
    # replace quantile values by categories
    for key, val in quantile_levels.items():
        vals = ge_levels[key].values
        vals = ['LOW' if x <= val[0.25] else 'NORMAL' if x > val[0.25] and x <= val[0.75] else 'HIGH' for x in vals]
        ge_levels[key] = vals

    # get gene occurences overall and from each patient
    occs, patient_occs = get_gene_occurences(relevance_scores, list(correct_patients), 200)

    generate_occurences(relevance_scores, concordance, subtypes, config)
    subtypes = {}
    if "net_cx_OccurenceInSubtype" in config.keys():
        subtypes = {key: val for val, key in enumerate(config["net_cx_OccurenceInSubtype"])}
    else:
        subtypes = {"CMS1": 0, "CMS2": 1, "CMS3": 2, "CMS4": 3, "Normal": 4}
    subtype_occs = count_subtype_occurence(subtypes, patient_occs, list(correct_patients), subtype_list, data.get_all_gene_names())

    node_attributes = {}
    for gene in used_genes:
        attrs = {}
        attrs["GE_Mean"] = gene_mean[gene]
        attrs["GE_Q25"] = quantile_levels[gene][0.25]
        attrs["GE_Q50"] = quantile_levels[gene][0.50]
        attrs["GE_Q75"] = quantile_levels[gene][0.75]
        attrs["GE_StdDev"] = stdev[gene]
        if gene in occs.keys():
            attrs["Occurrence"] = occs[gene]
        else:
            attrs["Occurrence"] = 0
        attrs["OccurenceBySubtype"] = subtype_occs[gene]
        # placeholder
        attrs["pvalue"] = 0.01
        attrs["qvalue"] = 1.0
        for patient in correct_patients:
            if gene not in patient_occs[patient]:
                continue                
            attrs[patient + "_GE"] = patient_ge[gene][patient]
            attrs[patient + "_GE_Level"] = ge_levels[gene][patient]
            attrs[patient + "_Score"] = relevance_scores[gene][patient]
        node_attributes[gene] = attrs
    # add metaRelSubnetVis specifics
    oa_keys = [x for x in list(config.keys()) if "metarelsubnetvis" in x]

    # TODO: adjust threshold properties
    opaque_aspects = {}
    attrs = {}
    for k in oa_keys:
        attrs[k[17:]] = config[k]
    # get thresholds
    #print("THRESHOLDS")
    #print(len(correct_patients))
    #tmp = data.feature_values.iloc[:, :-1]
    #print(np.max(np.max(tmp).values))
    #print(np.min(np.min(tmp[tmp > 0.0]).values))
    #tmp = relevance_scores[relevance_scores > 0.0]
    #print(np.nanmax(np.max(tmp).values))
    #print(np.nanmin(np.min(tmp).values))

    opaque_aspects["metaRelSubNetVis"] = [attrs]

    # save locally
    # upload network
    net_cx = ndex.generate_cx(data, relevance_scores, used_genes, network_attributes, node_attributes, opaque_aspects, config["path_to_results"] + "outp_graph.json")
