# glrp.postprocessing package

## Submodules

## glrp.postprocessing.gcnn_explainer module


### _class_ glrp.postprocessing.gcnn_explainer.GCNNExplainer(conf, model, data, mode='uniform')
Bases: [`BaseExplainer`](glrp.base.md#glrp.base.base_explainer.BaseExplainer)

The GCNExplainer from the paper:
> [ Explaining decisions of graph convolutional neural networks: patient-specific molecular subnetworks responsible for metastasis prediction in breast cancer]([https://pubmed.ncbi.nlm.nih.gov/33706810/](https://pubmed.ncbi.nlm.nih.gov/33706810/))
> Hryhorii Chereda, Annalen Bleckmann, Kerstin Menck, Júlia Perera-Bel, Philip Stegmaier, Florian Auer, Frank Kramer, Andreas Leha, Tim Beißbarth

The model can be used to explain the predictions for a graph of a graph convolutional neural network using
the proposed GLRP approach to work with the spektral framework. Original implementation: [https://gitlab.gwdg.de/UKEBpublic/graph-lrp](https://gitlab.gwdg.de/UKEBpublic/graph-lrp).

[https://mdpi-res.com/d_attachment/sensors/sensors-21-04536/article_deploy/sensors-21-04536-v2.pdf?version=1625210886](https://mdpi-res.com/d_attachment/sensors/sensors-21-04536/article_deploy/sensors-21-04536-v2.pdf?version=1625210886)


#### assign_rules_to_layers()
Assign a suiting LRP-rule to each model layer.


#### explain()
Explain a given model.


#### explain_graph()

#### prop_chebconv(a, w, R, polynomials, rule=LRPRule.EPSILON, first_layer=False)
Propagate outputs through a cheb conv layer.


#### prop_dense(a, w, R, rule=LRPRule.EPSILON)
Propagate outputs through a dense layer.
:params:

> a: vector of lower-layer activations
> layer: copy of the current layer
> R: relevances


#### prop_flatten(a, R)
Propagate outputs through a flatten layer.


#### prop_glob_max_pool(a, R)

#### prop_pool(a, R, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', type='max')
Propagate outputs through a max pooling layer.


#### rho(w, rule=LRPRule.EPSILON)

#### subgraph_to_cx()
Convert the resulting subgraphs to the cx format
to prepare it for submission to NDEx.
:return:

> net_cx: cx graph


### _class_ glrp.postprocessing.gcnn_explainer.LRPRule(value)
Bases: `Enum`

An enumeration.


#### DEFAULT(_ = _ )

#### EPSILON(_ = _ )

#### GAMMA(_ = _ )
## glrp.postprocessing.results module


### glrp.postprocessing.results.count_subtype_occurence(patient_occs, patient_list, subtype_list, gene_list)
Return the given occurences grouped by the patients subtypes.


### glrp.postprocessing.results.create_relevance_score_csv(relevances, data, output_path='relevance_scores.csv')
Creates a csv file with the relevances scores
mapped to the given patient ids and ppi genes


### glrp.postprocessing.results.generate_occurences(relevance_scores, concordance, subtypes, config)
Generate gene occurences by patient subtypes.


### glrp.postprocessing.results.generate_subnetwork_results(data, config, relevance_scores, used_genes, concordance, subtypes=None, output_path=None)
Generate ndex and metarelsubnetvis compatible results for the given network and data.


### glrp.postprocessing.results.get_gene_occurences(relevances, correct_patients, N)
Retrieve the N most selected genes from the given
relevance scores and a list of patients.
:returns: {Gene: #occ_num}

> 
> * dictionary: {Patient: [N_most_relevant_genes]}


* **Return type**

    
    * dictionary




### glrp.postprocessing.results.get_occurence_by_label(relevance_scores, cancer_patients, non_cancer_patients, N=20)
Retrieve top occurences by patient labels


### glrp.postprocessing.results.get_occurence_by_subtype(relevances, subtype_list, patient_list, N=20)
Retrieve occurences by top patient occurences
Warning: subtype_list and patient_list must have the correct alignment


### glrp.postprocessing.results.get_patient_names(data)
Return all sample names from the given data object.


### glrp.postprocessing.results.get_patient_subtype(patient_name, subtypes)
Get a subtype for a given patient name.


### glrp.postprocessing.results.get_subtype_information(patients, subtypes)
Get the patients grouped by subtypes.


### glrp.postprocessing.results.group_patients_by_subtype(subtype_list, patient_list)
Return the patients grouped by subtypes from the
given subtype and patient list (must be aligned).

## glrp.postprocessing.validation module


### glrp.postprocessing.validation.wgcna(f_path, output_fname)
Call the R wgcna script and perform
wgcna on the input data and save it to the given
output path.

## Module contents
