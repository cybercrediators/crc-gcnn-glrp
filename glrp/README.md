# Graph Layer-wise Relevance Propagation (GLRP)
This is an implementation of Layer-wise Relevance Propagation (LRP) for [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://arxiv.org/abs/1606.09375) based on [Spektral](https://graphneural.network/).
The code here is devoted to the paper "Explaining decisions of Graph Convolutional Neural Networks: patient specific molecular sub-networks responsible for metastasis prediction in breast cancer". The folder *lib* contains modifed code from Michaël Defferrard's [Graph CNN](https://github.com/mdeff/cnn_graph) repository. This tool will preprocess, train and visualize its Layer-wise relevance propagation all-in-one and generating, uploading and visualiasing them on NDEX and MetaRelSubnetVis.

## How To
### General Information
+ Folder structure:
  + `data/`: existing data e.g. used geo codes, mapped datasets, dataset labels etc.
  + `results/`: existing results e.g. relevance scores, saved models, saved concordances
  + `log/`: tensorboard logs of training metrics
  + `doc/`: Performance write-ups, dataset information
    + Validation R-Markdown-Skripte mit DEG und WGCNA Beispiel
  + `glrp/docs/`: HTML code documentation (sphinx)
  + `glrp/config/`: Tool configuration folder e.g. sample config, used configs

### Prerequisities
There should be *only* 3 steps to get everything going.
+ Create an appropriate config file (e.g. copy an existing one and adjust it). Most important parameters are (those folders will be created automatically):
  + `path_to_data`: folder of your data output csvs (e.g. `/path/to/data/[name]/`)
  + `path_to_results`: folder of your result output e.g. model
  + `path_to_log`: folder of your tensorboard metric outputs
  + `base_data_folder`: *should* be a subfolder of data for the raw CEL data (e.g. `/path/to/data/[name]/data`)
  + Various csv names e.g. `test_preprocessed.csv` -> names of the output csvs instead of default names 
  + `model_name`: name of your saved model
  + `net_cx_*`: All parameters with this prefix will be put into the generated CX model parameters e.g. `net_cx_name`
    + `net_cx_OccurenceInSubtype`: Also used to generate subtype occurence results
  + `ndex_username`/`ndex_password`: needed to upload the generated results to [NDEX](https://ndexbio.org)
  + `metarelsubnetvis_*`: All parameters with this prefix will be put in the generated CX model additional data fields
  + `visualisation_url`: Base link to MetaRelSubnetVis
+ Put your GEO Accession codes in a single `codes.txt` in the `path_to_data` (or you will be asked for a path)
  + In general every supported [annotation Package](https://bioconductor.org/packages/release/data/annotation/) for Affymetrix data is supported (maybe you have to add the according package to the `preprocessing.R` script) and `Affymetrix PrimeView` (you have to download the annotations from the affymetrix website and build it yourself)
  + Should be `.CEL`-raw data and Affymetrix data
+ Put your `labels.txt` with a list of Patient codes (you want to use) and label in the data folder
+ Continue with the command line tool

### Command line interface
+ Options:
  + `-h, --help`: show this help message and exit
  + `-p, --preprocessing`: Perform the necessary preprocessing steps on the data folder from the config file.
  + `--skip-scraper`: Skip the scraping part if you already downloaded and unpacked the gene expression data.
  + `--skip-r-preprocessing`: Skip the Preprocessing steps in R.
  + `-f, --full`: Full execution from preprocessing to training
  + `-t, --train`: Train a model from the given data in the config file
  + `--predef-patients PREDEF_PATIENTS`: Create a test set from a predefined list of patients
  + `--test-seed TEST_SEED`: Create a test set from a predefined test seed
  + `--save-concordance`: Save the concordance from the given model and data in the config file
  + `--used-genes USED_GENES`: Provide a txt list of genes to use in the result graph
  + `-o PREDICT [PREDICT ...]`: --predict the given patient data on an existing model
  + `--get-stats`: Get stats about data, model etc.
  + `-e, --explain`: Explain the given data and the config file. For single data prediction/explanation use the predict option.
  + `-v, --visualize`: Generate results and upload the generated results to ndex.
  + `-c C, --config C`: Define config file
+ Examples:
  + `python glrp.py -f -c /path/to/config/file` - Execute all steps with the given config
  + `python glrp.py -t -c /path/to/config/file --test-seed 42` - ONLY Train a model with a split random seed 42 (and the given config file)
  + `python glrp.py -t -c /path/to/config/file --predef-patients /path/to/patient/names.txt` - ONLY Train a model with a given txt list of patients to make up the test set (and the given config file)
  + `python glrp.py -p -c /path/to/config/file` - ONLY preprocess the data in the data folder (and the given config file)
  + `python glrp.py -p -c /path/to/config/file --skip-scraper` - ONLY preprocess the data, but don't scrape any GEO codes (and the given config file)
  + `python glrp.py -e -c /path/to/config/file` - ONLY perform the glrp on the given model/data (and the given config file)
  + `python glrp.py -v -c /path/to/config/file` - ONLY generate results for the given model/data (and the given config file)

## More Information
### Data
Various datasets were analyzed with this tool, namely the original breast cancer dataset (969 patients, 567 without metastasis/393 without metastasis) and two colorectal cancer datasets (733 samples, 303 normal/434 cancer | 1400 X/Y) More details, e.g. the used GEO accession codes, can be found in the [corresponding data documentation](../doc/data.md).

### Preprocessing
The preprocessing is explained in the [corresponding preprocessing documentation](../doc/preprocessing.md). In short, on the raw data of each microarray the RMA-algorithm will be applied. Then all RMA preprocessed data will be combined and quantile normalized. Then the cancer subtypes will be predicted for all patients, using the [CMScaller R-package](https://github.com/peterawe/CMScaller).

### Networks
After preprocessing the input data, they'll be mapped onto a given network. This will be obtained and converted from the [NDEX platform](https://ndexbio.org/). The used data model is briefly explained in the [corresponding documentation](../doc/network.md)

### GCNN
The used GCNN model uses the [spektral](https://graphneural.network) implementation of the ChebNet Convolutional layer. More information about graph convolutional neural networks can be found in the [documentation](../doc/gcn.md).

![GCNN Architecture](../text/master/Data/diagrams/neural_net.png)

### GLRP/LRP
Specific information about how layer-wise relevance propagation works can be found [here](../doc/lrp.md). Also you can find more information on the used graph layer-wise relevance propagation [documentation](../doc/glrp.md).

### Results
Generated results e.g. performance or visualisation can be found in the [corresponding document](../doc/results.md).
