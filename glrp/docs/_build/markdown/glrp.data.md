# glrp.data package

## Submodules

## glrp.data.data_model module


### _class_ glrp.data.data_model.DataModel(config)
Bases: `object`

Define a data model and helper functions for the spektral dataset


#### coarsen_inputs(num_layers, adj_mat)
Preprocess inputs and coarsen the graphs (according
to the original proposal)


#### get_all_gene_names()
Retrieve and return all gene names from the features.


#### init_data()
Initialize the data from the given directories from
the config file.


#### show_data_infos()
Show information about the processed data from this data class


#### train_test_split(size, predefined_test_patients=None, seed=None)
create a train and test split from input data


#### train_test_val_split(test_size, val_size)
create a train, validation and test split from input data

## glrp.data.ge_dataset module


### _class_ glrp.data.ge_dataset.GeDataset(labels, feature_vals, feature_graph, normalize, perm=None, \*\*kwargs)
Bases: `Dataset`

Dataset for crc data


#### read()
Read the input features and feature graph and return a spektral dataset

## glrp.data.ndex module


### _class_ glrp.data.ndex.NdexManager(config)
Bases: `object`

Up-/Download networks from ndex and manage them to use with glrp


#### download_network(net_id)
Retrieve ndex network and return networkx object


#### generate_cx(data, rel_scores, used_genes, network_attributes, node_attributes=None, opaque_aspects=None, output_path=None)
Create nicecx network from the given relevance scores
for each patient.
- (opt.) save the raw data to the given output path
- return the created cx network


#### generate_metarelsubnetvisURL(uuid)
show the corresponding metarelsubnetvis url (local/remote)


#### save_adj(output_path, net_id)
Download and convert a given network id from ndex and save it to the given
output_path.


#### save_network(outputPath, net_id)
Download a given network and save to the given output file


#### upload_network(net_cx, visibility='PRIVATE')
upload a given cx network back to onto the ndex platform

## Module contents
