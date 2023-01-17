# glrp.preprocessing package

## Submodules

## glrp.preprocessing.labels module


### glrp.preprocessing.labels.generate_labels_from_preprocessed(preprocessed_fname, labels_fname, outp_fname)
Generate a label file from the preprocessed gene expressions and the given
label list

## glrp.preprocessing.mapping module


### glrp.preprocessing.mapping.map_preprocessed_to_gene_list(preprocessed, genes)
Map preprocessed gene expressions onto a list of genes and
return the resulting dict.


### glrp.preprocessing.mapping.map_preprocessed_to_graph(preprocessed_path, graph_path, output_file)
Map a given preprocessed file onto a given network graph and write
the output to the given output file name.


### glrp.preprocessing.mapping.write_output(fname, data)
write to mapped csv file

## glrp.preprocessing.preprocess_r module

## glrp.preprocessing.rbridge module


### _class_ glrp.preprocessing.rbridge.RBridge(script_path, function_name)
Bases: `object`

Call and use functions from existing
R scripts


#### call(\*args)
## glrp.preprocessing.scrape module


### _class_ glrp.preprocessing.scrape.DownloadProgress(\*_, \*\*__)
Bases: `tqdm`


#### update_to(b=1, bsize=1, tsize=None)

### _class_ glrp.preprocessing.scrape.GeoScraper(config)
Bases: `object`


#### build_url()

#### clean_raw_files()
remove downloaded raw tar files from the base data folder


#### download(fpath, url)

#### download_code(code)
Download a given geo accession code into a subfolder of a given name


#### download_file(fname)
Download geo datasets from a given txt file.


#### full_prepare_data(codes_fname)
Download and prepare RAW data for further preprocessing.


#### read_codes(fname)
read gse codes from a filename and return them


#### unpack_cel(src, dest)
Unpack a given .cel.gz file


#### unpack_files()
Unpack all downloaded files in the current data folder


#### unpack_tar(fname)
Unpack a given .tar archive.

## glrp.preprocessing.subclass_prediction module

## Module contents
