
# get output data

subtype_prediction <- function(dirname) {
  #' colorectal cancer subtype prediction
  #' @param dirname -> directory name of the affy files

  ### dependencies: run if not already installed
  ### limma has lof of dependencies - takes time
  #source("https://bioconductor.org/biocLite.R")
  BiocManager::install(c("Biobase", "limma"))
  install.packages("devtools")
  
  ### install: latest version
  devtools::install_github("Lothelab/CMScaller")
  
  library(Biobase)
  library(CMScaller)
  library(affy)
  par(mfrow=c(1,2))
  
  # get input data
  data <- ReadAffy()
  inp_data <- 0
  # classify subtypes with cmscaller, input data was already preprocessed
  res <- CMScaller(inp_data, RNAseq = FALSE)
}
