if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")

BiocManager::install("limma")
BiocManager::install("affy")
BiocManager::install("SpikeInSubset")
BiocManager::install("annotate")
BiocManager::install("lumi")


library(affy)
library(SpikeInSubset)
library(limma)
library(annotate)



perform_deg <- function(folder_path) {
  # read data
  #setwd(folder_path)
  to_preprocess = list.files(path = folder_path, full.names = TRUE, pattern = "\\.CEL$")
  preprocess_data = ReadAffy(filenames = to_preprocess)
  
  # apply rma probe summary, quantile normalization
  rma_normalized <- rma(preprocess_data, normalize=TRUE)
  eset <- exprs(rma_normalized)
  pdat <- pData(rma_normalized)
  
  # 
}


# folder_path <- "/home/lutzseba/Desktop/ma/GPL10558_HumanHT-12_V4_0_R2_15002873_B.txt"
folder_path <- "/home/lutzseba/Desktop/ma/GSE8671/"
perform_deg(folder_path)