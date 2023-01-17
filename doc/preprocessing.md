# Colorectal cancer
## Preprocessing
+ Infos: https://www.frontiersin.org/articles/10.3389/fonc.2017.00135/full
+ RMA-probe summary: https://academic.oup.com/biostatistics/article/4/2/249/245074
  + preprocessing for Affymetrix data
  + background correction
  + log2 transform
  + quantile normalization
  + linear model for normalization to obtain expression measure for each probe set
  + used r rma package: https://www.rdocumentation.org/packages/affy/versions/1.50.0/topics/rma
+ RMA is applied on every dataset individually
+ All preprocessed datasets are then combined
+ Quantile normalization then is applied onto the combined set of all individually preprocessed datasets
+ (avoid batch effect)
+ Integrated in the framework via the [rpy2](https://rpy2.github.io/) bridge

## Subtype Prediction
+ subtype prediction:
  + https://www.sciencedirect.com/science/article/pii/S0923753419311317?via%3Dihub
  + https://www.nature.com/articles/s41598-020-69083-y
  + https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4636487/
+ cmscaller: https://github.com/peterawe/CMScaller
  + corresponding paper: https://www.nature.com/articles/s41598-017-16747-x#Sec7
+ use pre-processed .CEL files from GEO as input
+ let cmscaller detect crc subtypes

## Mapping
+ Take an existing graph (in this case HPRD PPI)
  + download and convert an existing network from the NNEx database
  + use the output adjacency matrix
+ map genes from the preprocessed dataset to the genes in the graph
  + take existing genes from the graph
  + obtain the genes from the preprocessed data
  + create a new dataframe and set the corresponding probe values for each found gene
  + if multiple probes are mapping to one gene, use probe with the highest avg. value

## Labels
+ create label files by hand from the corresponding dataset papers
+ map labels to the preprocessed dataset with `create_labels.py`
+ Labels: cancer: 0/1
  + classify adenoma datasets individually
  + TODO: find other labeling technique

## Validation(?)
+ Validation (HUVECs before/after TNFalpha): (from original: https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0230884&type=printable)
