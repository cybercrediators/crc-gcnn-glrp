install.packages("affy", repos = "http://cran.us.r-project.org")
BiocManager::install(c("SpikeInSubset", "hgu133plus2.db", "limma", "annotate"))
BiocManager::install(c("Biobase", "limma", "devtools"))

library(affy)
library(SpikeInSubset)
library(limma)
library(annotate)

# annotations for hgu133plus
library(hgu133plus2.db)
library(hgu133a.db)
library(hgu219.db)
library(hthgu133a.db)

devtools::install_github("Lothelab/CMScaller")

library(CMScaller)

predict_subtypes <- function(preprocessed_data_path, output_path) {
  # TODO: correct eset reader
  # read all cel files from the data directory
  par(mfrow=c(1,2))
  
  to_predict = list.files(path = preprocessed_data_path, full.names = TRUE, pattern = "\\.CEL$", recursive = TRUE)
  to_predict
  
  to_predict_affy <- ReadAffy(filenames = to_predict)
  
  eset <- rma(to_predict_affy, normalize=TRUE)
  rma_eset <- exprs(eset)
  genenames <- rownames(rma_eset)
  geneSymbols <- getSYMBOL(genenames, annotation(eset))
  symbolNames <- as.vector(geneSymbols)
  
  emat <- data.frame(rma_eset)
  row.names(emat) <- make.names(c(geneSymbols), unique = TRUE)
  
  subtypes <- CMScaller(emat, RNAseq = FALSE, rowNames = "symbol", doPlot = FALSE)
  write.csv(subtypes, file = output_path)
}


# folder_path <- "/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/crc/data/"
# output_path <- "/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/crc/data/subtypes.csv"
# 
# predict_subtypes(folder_path, output_path)
# 
# base_folder <- "/home/lutzseba/Desktop/ma/gcnn-and-grlp/data/crc_extended/data/"
# 
# dirs <- list.dirs(path = base_folder, full.names = TRUE, recursive = FALSE)
# dirs
# 
# for (gse in dirs) {
#   folder_path <- paste(gse, "/", sep="")
#   output_path <- paste(folder_path, "subtypes.csv", sep="")
#   print(folder_path)
#   print(output_path)
#   # folder_path <- paste(base_folder, gse, "/", sep="")
#   # output_path <- paste(folder_path, gse, ".csv", sep="")
#   predict_subtypes(folder_path, output_path)
# }
