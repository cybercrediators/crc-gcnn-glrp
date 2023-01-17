import json
import sys
import ndex2
import networkx as nx
import pandas as pd
import numpy as np
import copy
from networkx.readwrite import json_graph


class NdexManager(object):

    """Up-/Download networks from ndex and manage them to use with glrp"""

    def __init__(self, config):
        self.config = config
        # self.graph_path = config["base_data_path"] + "graphs/"

    def save_adj(self, output_path, net_id):
        """
        Download and convert a given network id from ndex and save it to the given
        output_path.
        """
        net_cx, g = self.download_network(net_id)
        # create an adjacency matrix from the converted networkx graph
        A = nx.to_numpy_array(g)
        adj = np.maximum(A, A.transpose())
        # print(adj)
        cols = {}
        for id, node in net_cx.get_nodes():
            node_name = node.get('n')
            cols[id] = node_name
        # adj = nx.to_pandas_adjacency(g)
        adj = pd.DataFrame(adj)
        # adj = adj.astype(float)
        adj = adj.astype(int)
        adj = adj.rename(columns=cols)
        adj.to_csv(output_path, index=False)

    def download_network(self, net_id):
        """
        Retrieve ndex network and return networkx object
        """
        # get cx file from ndex
        client = ndex2.client.Ndex2()
        client_res = client.get_network_as_cx_stream(net_id)

        # create ndex2 tool nice cytoscape format
        net_cx = ndex2.create_nice_cx_from_raw_cx(json.loads(client_res.content))
        # print('Name: ' + net_cx.get_name())
        # print(json.dumps(net_cx.to_cx())[0:100])

        # convert cytoscape network to python networkx
        g = net_cx.to_networkx(mode='default')

        # print('Name: ' + str(g))
        # print('Number of nodes: ' + str(g.number_of_nodes()))
        # print('Number of edges: ' + str(g.number_of_edges()))
        # print('Network annotations: ' + str(g.graph))
        return net_cx, g

    def save_network(self, outputPath, net_id):
        """
        Download a given network and save to the given output file
        """
        net_cx, g = download_network(net_id)
        with open(outputPath, 'w', encoding='utf-8') as f:
            json.dump(net_cx.to_cx(), f, ensure_ascii=False, indent=4)

    def generate_metarelsubnetvisURL(self, uuid):
        """show the corresponding metarelsubnetvis url (local/remote)"""
        pass

    def generate_cx(self, data, rel_scores, used_genes, network_attributes, node_attributes=None, opaque_aspects=None, output_path=None):
        """
        Create nicecx network from the given relevance scores
        for each patient.
        - (opt.) save the raw data to the given output path
        - return the created cx network
        """
        # create feature graph
        gene_list = data.get_all_gene_names()
        to_remove = [x for x in gene_list if x not in used_genes]
        # print(to_remove)
        graph = copy.deepcopy(data.feature_graph)
        #print(list(graph.index.values))
        #print(list(graph.columns))
        graph = graph.rename(index=dict(zip(list(graph.index.values), list(graph.columns))))
        #print(graph)

        # remove unused genes/nodes and corresponding edges
        nx_graph = nx.from_pandas_adjacency(graph)
        nx_graph.remove_nodes_from(to_remove)

        for (_, _, d) in nx_graph.edges(data=True):
            d.clear()
        
        #net_cx = ndex2.create_nice_cx_from_pandas(raw_graph)
        net_cx = ndex2.create_nice_cx_from_networkx(nx_graph)

        # add node attributes
        if node_attributes:
            nodes = net_cx.get_nodes()
            for id, node in nodes:
                name = node.get('n')
                if name in node_attributes:
                    for key, val in node_attributes[name].items():
                        dtype = 'double'
                        if (type(val) is str):
                            dtype = 'string'
                        if (type(val) is list):
                            dtype = 'list_of_string'
                        net_cx.set_node_attribute(id, key, val, dtype, overwrite=True)

        # add opaque aspects e.g. metarelsubnetvis
        if opaque_aspects:
            for key, val in opaque_aspects.items():
                net_cx.set_opaque_aspect(key, val)
        
        # add network attributes
        for attr in network_attributes:
            dtype = 'string'
            if (type(attr[1]) is list):
                dtype = 'list_of_string'
            net_cx.set_network_attribute(attr[0], attr[1], dtype)

        self.upload_network(net_cx, visibility="PUBLIC")

        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(net_cx.to_cx(), f, ensure_ascii=False, indent=4)
        return net_cx


    # load back to ndex (https://github.com/ndexcontent/ndexncipidloader)
    def upload_network(self, net_cx, visibility='PRIVATE'):
        """
        upload a given cx network back to onto the ndex platform
        """
        if not all (k in self.config.keys() for k in ("ndex_password","ndex_username")):
            print("No password/username for ndex was found!")
        # Create client, be sure to replace <USERNAME> and <PASSWORD> with NDEx username & password
        try:
            client = ndex2.client.Ndex2(username=self.config["ndex_username"], password=self.config["ndex_password"])
            # Save network to NDEx, value returned is link to raw CX data on server.
            res = client.save_new_network(net_cx.to_cx(), visibility=visibility)
            if (res):
                print("Your network was uploaded successfully!")
                print(res)
                print("\n MetaRelSubnet visualisation: \n " + self.config["visualisation_url"] + "?" + "uuid=" + res.split("/")[-1])
        except Exception as e:
            print("Your network couldn't be uploaded!")
            raise e


#uuid = "079f4c66-3b77-11ec-b3be-0ac135e8bacf"
# nd = NdexManager({})
# nd.save_adj("test.csv", "275bd84e-3d18-11e8-a935-0ac135e8bacf")
