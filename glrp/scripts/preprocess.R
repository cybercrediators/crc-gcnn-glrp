

preprocess <- function(dirname) {
  #' Preprocessing for geo files
  #' @param dirname -> name of geo file directory 

  install.packages("affy", repos = "http://cran.us.r-project.org")
  BiocManager::install(c("SpikeInSubset", "hgu133plus2.db", "limma", "annotate"))
  library(SpikeInSubset)
  library(affy)
  library(limma)
  library(annotate)
  # annotations for hgu133plus
  library(hgu133plus2.db)
  #library(sva)
  
  
  # read files
  Data <- ReadAffy()
  # log2 transform
  #Data <- log(Data)
  slotNames(Data)
  sampleNames(Data)
  annotation(Data)
  
  # perform rma probe summary, quantile normalization
  eset <- rma(Data, normalize=TRUE)
  rma_eset <- exprs(eset)
  exprs(eset)
  pData(eset)
  pdat <- pData(eset)
  dim(eset); class(eset)
  #plot(rowMeans(rma_eset), pch=".",  main="Scatter plot")
  a <- rowMeans(rma_eset)
  #plot(a)
  
  # remove batch effect(?)
  #strain <- c(1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0)
  #design <- model.matrix(~factor(strain))
  #colnames(design) <- c("CC", "N")
  #design
  #dim(eset)
  #dim(design)
  
  #fit <- lmFit(eset, design)
  #ebayes <- eBayes(fit)
  #names(ebayes)
  #topTable(ebayes, coef=2)
  
  # annotations
  annotation(eset)
  rownames(rma_eset)
  genenames <- rownames(rma_eset)
  #geneID <- getEG(genenames, "hgu133plus2.db")
  geneID <- getEG(genenames, annotation(eset))
  geneSymbols <- getSYMBOL(genenames, annotation(eset))
  
  geneID[1:5]
  geneSymbols[1:5]
  
  
  # print gene expressions
  test <- data.frame(geneSymbols, rma_eset)
  res <- data.frame(geneID, test)
  res[1:5, ]
  # output to file
  write.csv(res, file="test_preprocessed.csv")
}
