BiocManager::install(c("SpikeInSubset", "hgu133plus2.db", "limma", "annotate"))
BiocManager::install("affy")
BiocManager::install(c("Biobase", "limma", "hgu219.db"))
BiocManager::install("AnnotationForge")
BiocManager::install("human.db0")
BiocManager::install("pd.hta.2.0")
#BiocManager::install("hgu133a.db")
# BiocManager::install("hthgu133a.db")
# BiocManager::install("affycoretools")
# BiocManager::install("ht20probeset.db")


library(affy)
library(SpikeInSubset)
library(limma)
library(annotate)

#require(AnnotationForge)
# makeDBPackage('HUMANCHIP_DB',
#               affy = TRUE,
#               prefix = 'primeview',
#               fileName = '/home/lutzseba/Desktop/ma/PrimeView.na36.annot.csv',
#               baseMapType = 'eg',
#               outputDir = '.',
#               author = 'Bioconductor',
#               version = '0.99.1',
#               manufacturer = 'Affymetrix',
#               manufacturerUrl = 'http://www.affymetrix.com')
# install.packages('primeview.db', repos = NULL, type = 'source')
# library(primeview.db)
# available.dbschemas()
# makeDBPackage('HUMANCHIP_DB',
#               affy = TRUE,
#               prefix = 'pd.hta.2.0',
#               fileName = '/home/lutzseba/Desktop/ma/HTA-2_0.na36.hg19.transcript.csv',
#               baseMapType = 'eg',
#               outputDir = '.',
#               author = 'Bioconductor',
#               version = '0.99.1',
#               manufacturer = 'Affymetrix',
#               manufacturerUrl = 'http://www.affymetrix.com')
# install.packages('pd.hta.2.0.db', repos = NULL, type = 'source')
# library(primeview.db)

# annotations for hgu133plus
library(hgu133plus2.db)
# library(hgu133a.db)
# library(hgu219.db)
# library(hthgu133a.db)
# library(pd.hta.2.0.db)
# library(affycoretools)

preprocess_folder <- function(folder_path, output_path) {
  # read data and set working directory
  #setwd(folder_path)
  to_preprocess = list.files(path = folder_path, full.names = TRUE, pattern = "\\.CEL$")
  preprocess_data = ReadAffy(filenames = to_preprocess)
  
  #preprocess_data <- oligo::read.celfiles(to_preprocess)
  # apply rma probe summary, quantile normalization
  eset <- rma(preprocess_data, normalize=TRUE)
  eset
  
  rma_eset <- exprs(eset)
  pdat <- pData(eset)
  rma_eset
  annotation(eset)
  # annotate gene symbols
  genenames <- rownames(rma_eset)
  geneID <- getEG(genenames, annotation(eset))
  geneSymbols <- getSYMBOL(genenames, annotation(eset))
  
  
  test <- data.frame(geneSymbols, rma_eset)
  res <- data.frame(geneID, test)
  write.csv(res, file=output_path)
}
# 

preprocess_single <- function(fname, output_path) {
  preprocess_data = ReadAffy(filenames = fname)
  
  #preprocess_data <- oligo::read.celfiles(to_preprocess)
  # apply rma probe summary, quantile normalization
  eset <- rma(preprocess_data, normalize=TRUE)
  eset
  
  rma_eset <- exprs(eset)
  pdat <- pData(eset)
  rma_eset
  annotation(eset)
  # annotate gene symbols
  genenames <- rownames(rma_eset)
  geneID <- getEG(genenames, annotation(eset))
  geneSymbols <- getSYMBOL(genenames, annotation(eset))
  
  
  test <- data.frame(geneSymbols, rma_eset)
  res <- data.frame(geneID, test)
  write.csv(res, file=output_path)
}
# 
# fname <- "/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/test/single_pred/GSM215080.CEL"
# outp <- "/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/test/single_pred/GSM215080.csv"
# preprocess_single(fname, outp)

# base_folder <- "/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/crc_extended/data/"
# 
# folder_path <- "/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/test/data/GSE8671/"
# output_path <- "/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/test/data/GSE8671/GSE8671.csv"
# preprocess_folder(folder_path, output_path) 
# gse_list <- list("GSE44076", "GSE44861", "GSE41258", "GSE81558", "GSE113513", "GSE110224")
# for (gse in gse_list) {
#   folder_path <- paste(base_folder, gse, "/", sep="")
#   output_path <- paste(folder_path, gse, ".csv", sep="")
#   preprocess_folder(folder_path, output_path)
# }
# 
# gse_list =  readLines("/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/crc_extended/data/codes.txt")
# #gse = '73360'
# 
# gse = 'GSE100179'
# folder_path <- paste(base_folder, gse, "/", sep="")
# output_path <- paste(folder_path, gse, ".csv", sep="")
# 
# 
# # read data and set working directory
# #setwd(folder_path)
# to_preprocess = list.files(path = folder_path, full.names = TRUE, pattern = "\\.CEL$")
# #preprocess_data = ReadAffy(filenames = to_preprocess)
# 
# preprocess_data <- oligo::read.celfiles(to_preprocess)
# # apply rma probe summary, quantile normalization
# eset <- oligo::rma(preprocess_data, normalize=TRUE)
# eset
# 
# rma_eset <- exprs(eset)
# pdat <- pData(eset)
# rma_eset
# 
# 
# annotation(eset)
# # annotate gene symbols
# genenames <- rownames(rma_eset)
# geneID <- getEG(genenames, annotation(eset))
# geneSymbols <- getSYMBOL(genenames, annotation(eset))
# 
# 
# test <- data.frame(geneSymbols, rma_eset)
# res <- data.frame(geneID, test)
# write.csv(res, file=output_path)