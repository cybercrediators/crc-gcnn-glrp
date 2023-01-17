# Networks
## HPRD PPI
+ HPRD PPI network (https://academic.oup.com/nar/article/37/suppl_1/D767/1019294)

# Ndex/CX data model
+ network attributes
  + name-value pair as annotation
+ nodes
  + each node has an ID, represents a node in the network
  + can also specify node name, standard identifier or both
+ edges
  + specify edges connecting nodes
  + has an ID and connected node IDs
  + (opt) node interaction
+ nodeAttributes
  + name-value pair describing one node (ID)
+ edgeAttributes
  + name-value pair describing one edge (ID)

# Ndex2 python
+ use https://ndex2.readthedocs.io/en/latest/installation.html
+ convert downloaded cytoscape to networkx
+ use networkx to create a pandas adjacency matrix
+ set correct node column names
