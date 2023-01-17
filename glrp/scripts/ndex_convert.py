import json
import sys
import ndex2
import networkx as nx
import pandas as pd
import numpy as np
from networkx.readwrite import json_graph

## INPUT: ndex uuid
#
#if len(sys.argv) != 3:
#    print("Usage: ndex_convert.py [UUID] [OUTPUT_FILENAME]")
#    sys.exit()
#UUID=sys.argv[1]
#FNAME=sys.argv[2]
##UUID='275bd84e-3d18-11e8-a935-0ac135e8bacf'
#
## get cx file from ndex
#client = ndex2.client.Ndex2()
#client_res = client.get_network_as_cx_stream(UUID)
#
## create ndex2 tool nice cytoscape format
#net_cx = ndex2.create_nice_cx_from_raw_cx(json.loads(client_res.content))
#print('Name: ' + net_cx.get_name())
#print(json.dumps(net_cx.to_cx())[0:100])
##with open(FNAME+'_cx.json', 'w', encoding='utf-8') as f:
##    json.dump(net_cx.to_cx(), f, ensure_ascii=False, indent=4)
#
## convert cytoscape network to python networkx
#g = net_cx.to_networkx(mode='default')
#
#print('Name: ' + str(g))
#print('Number of nodes: ' + str(g.number_of_nodes()))
#print('Number of edges: ' + str(g.number_of_edges()))
#print('Network annotations: ' + str(g.graph))
#
## convert networkx to adj. matrix
## get cx node annotations for the correct gene symbols
#cols = {}
#
#for id, node in net_cx.get_nodes():
#    node_name = node.get('n')
#    cols[id] = node_name
#
#
#A = nx.to_numpy_array(g)
#
## add weights to the adjacency_matrix of the string database with their score
##weights = {}
##for id, edge in net_cx.get_edges():
##    score = net_cx.get_edge_attribute_value(edge, 'combined_score')
##    x1 = edge.get('s')
##    x2 = edge.get('t')
##    weights[(x1, x2)] = float(score)
##
##scores = weights.values()
##max_ = max(scores)
##min_ = min(scores)
##norm_weights = {key: ((v - min_ ) / (max_ - min_) )  for (key, v) in weights.items() }
##for pos, val in norm_weights.items():
##    A[pos[0]][pos[1]] = val
#
#
#adj = np.maximum( A, A.transpose() )
#print(adj)
#
## create an adjacency matrix from the converted networkx graph
##adj = nx.to_pandas_adjacency(g)
#adj = pd.DataFrame(adj)
##adj = adj.astype(float)
#adj = adj.astype(int)
#adj = adj.rename(columns=cols)
#adj.to_csv(FNAME+'_adj.csv', index=False)


# load back to ndex (https://github.com/ndexcontent/ndexncipidloader)
# my_uuid = "5b1ba1bf-022b-11ed-ac45-0ac135e8bacf"
# orig_uuid = "a420aaee-4be9-11ec-b3be-0ac135e8bacf"
# client = ndex2.client.Ndex2()
# client_res = client.get_network_as_cx_stream(my_uuid)
# my_nw = ndex2.create_nice_cx_from_raw_cx(json.loads(client_res.content))
# #with open('my_cx.json', 'w', encoding='utf-8') as f:
# #    json.dump(net_cx.to_cx(), f, ensure_ascii=False, indent=4)
# #
# client_res = client.get_network_as_cx_stream(orig_uuid)
# orig_nw = ndex2.create_nice_cx_from_raw_cx(json.loads(client_res.content))
# # with open('orig_cx.json', 'w', encoding='utf-8') as f:
# #     json.dump(net_cx.to_cx(), f, ensure_ascii=False, indent=4)
# 
# orig_node_ids = {}
# for id, node in orig_nw.get_nodes():
#     orig_node_ids[node.get('n')] = id
# 
# for id, node in my_nw.get_nodes():
#     node_name = node.get('n')
#     orig_id = orig_node_ids[node_name]
#     orig_qval = orig_nw.get_node_attribute(orig_id, "qvalue")
#     orig_pval = orig_nw.get_node_attribute(orig_id, "pvalue")
#     print("ORIG: ", orig_id, orig_qval, orig_pval)
#     my_nw.set_node_attribute(id, "pvalue", orig_pval.get('v'), type="double", overwrite=True)
#     my_nw.set_node_attribute(id, "qvalue", orig_qval.get('v'), type="double", overwrite=True)
#     
# # with open('my_cx.json', 'w', encoding='utf-8') as f:
# #     json.dump(my_nw.to_cx(), f, ensure_ascii=False, indent=4)
# 
# new_net = my_nw.to_cx()
# 
# client = ndex2.client.Ndex2(username="lutzseba", password="3p4hnX8wv6x5VBBCZkodaiEln")
# res = client.save_new_network(new_net, visibility='PRIVATE')
# print(res)
