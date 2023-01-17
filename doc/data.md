# Data formats
+ `.CEL` files, created by affymetrix dna analysis software
+ contains data extraced from probes

# Input Data
+ HPRD PPI: protein-protein interaction network (graph) to structure gene expression data
  + (seems to be) adjacency matrix of the undirected graph
+ GEO_HG_PPI:
  + patient-specific gene expression data mapped onto the undirected graph
  + normalized
  + 969 patients (393 dist. metastasis, 576 without metastasis), 12179 genes
  + 207/206 connected components, 7168/6888 vertices
+ labels_GEO_HG:
  + 0/1 labels for each patient

# Which data is needed?
+ graph/network e.g. ppi, word-embeddings etc.
+ graph features e.g. cancer patient gene expression data
  + preprocessing
+ feature labels

## How data was obtained
+ 10 datasets from GEO
+ RMA probe-summary algorithm to process each dataset (https://doi.org/10.1093/biostatistics/4.2.249)
+ only samples with metadata on metastasis-free survival + quantile normalization
+ molecular subtype prediction using genefu R-package (https://www.bioconductor.org/packages/release/bioc/html/genefu.html)

## How network was obtained
+ HPRD PPI network (https://academic.oup.com/nar/article/37/suppl_1/D767/1019294)

# Results
+ patient-specific subnetwork constructed from the 140 most relevant genes

# Colorectal Cancer Data
## Info
+ Molecular subtypes (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6503627/) (check for cutoffs):
  + CMS1, immune
  + CMS2, canonical
  + CMS3, metabolic
  + CMS4, mesenchymal

## Datasets
+ Paper: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7563725/

## new final dataset
+ 1341 train/150 test samples
+ 1400/1438 samples
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE8671
  + 32, 32
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE9348
  + 70 crc, 12 normal
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE23878 (log2)
  + 35 crc, 24 normal
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE37364 (log2)
  + 94 samples , 38 normal, 56 crc
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE15960
  + 6 normal, 6 adenoma, 6 crc
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22598
  + 17 normal, 17 crc, rma applied
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4183
  + 15 crc, 15 adenoma, 15 ibd, 8 normal
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100179
  + 20 normal, 20 adenoma, 20 crc
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33113
  + 90 crc stage 2, 6 normal
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE18105
  + 111 samples, 77 crc, 17 normal, rma applied
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73360
  + 92/46 samples (duplicates), 19 normal
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20916
  + 145 samples, 105 macro, 40 microdissected
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4107
 + 22 samples, 10 healthy, 12 patients
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE24514
  + 49 samples, 34 crc, 15 normal
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44076
  + 246 samples, 98 tumor, 98 healthy, 50 healthy -> that's nice
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44861
  + 111 samples, non-cancerous + cancerous
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41258
  + 390 samples, various samples non-cancerous and cancerous
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81558
  + 32 crc, 9 non-cancerous
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE113513
  + 28 samples, 14/14 cancer/non-cancerous
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE110224
  + 34 samples, 17/17

Optional:
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39582
  + 19 non-tumoral, 443 cc, 123 cc validation (don't use everything maybe)

+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=gse106582
  + not affymetrix
  + 194 samples
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41657
  + not affymetrix
  + 12 normal, 21 low-grade adenomas, 30 high-grade adenomas, 25 adenocarcinomas

## Used test dataset:
+ 106: 0/normal, 132 1/crc
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE8671
  + 32, 32
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE9348
  + 70 crc, 12 normal
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE23878 (log2)
  + 35 crc, 24 normal
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE37364 (log2)
  + 94 samples , 38 normal, 56 crc

# Final dataset
+ 303: 0/normal, 434 1/crc, 733 samples
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE8671
  + 32, 32
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE9348
  + 70 crc, 12 normal
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE23878 (log2)
  + 35 crc, 24 normal
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE37364 (log2)
  + 94 samples , 38 normal, 56 crc
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE15960
  + 6 normal, 6 adenoma, 6 crc
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE22598
  + 17 normal, 17 crc, rma applied
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4183
  + 15 crc, 15 adenoma, 15 ibd, 8 normal
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE33113
  + 90 crc stage 2, 6 normal
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE18105
  + 111 samples, 77 crc, 17 normal, rma applied
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE73360
  + 92/46 samples (duplicates), 19 normal
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE20916
  + 145 samples, 105 macro, 40 microdissected
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE4107
 + 22 samples, 10 healthy, 12 patients
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE24514
  + 49 samples, 34 crc, 15 normal

## more possible datasets:
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13067
  + 74 colorectal cancer samples, MAS5.0 applied
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13294
  + 155 colorectal cancer samples, MAS5.0 applied
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE14333
  + 290 colorectal cancer samples, MAS5.0 applied
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE17536
  + 177 colorectal cancer patients
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE2109
  + multiple tumor samples, some from colon too
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE35896
  + 62 crc samples, rma applied
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE21510
  + 148 crc samples
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE32323
  + 17 normal, 17 crc, rma applied
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE117606
  + 70 patients, different samples
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE39582
  + 19 non-tumoral, 443 cc, 123 cc validation (don't use everything maybe)
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE68468
  + 390 samples, individual labels...

## even more datasets
+ GSE42284
  + 188 crc samples, Agilent G2565AA scanner
+ GSE37892
  + 130 crc samples, hgu133plus2
+ TCGA samples -> colorectal cancer -> tcga
  + https://portal.gdc.cancer.gov/
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44076
  + 246 samples, 98 tumor, 98 healthy, 50 healthy -> that's nice
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE44861
  + 111 samples, non-cancerous + cancerous
+ https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE41258
  + 390 samples, various samples non-cancerous and cancerous
