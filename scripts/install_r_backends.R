# Explicit installation helper. Analysis functions never run this script.
# Select a private library with R_LIBS_USER before invoking Rscript.
args <- commandArgs(trailingOnly=TRUE)
if (any(!args %in% "--extended")) stop("Usage: Rscript install_r_backends.R [--extended]")
options(repos=c(CRAN="https://cloud.r-project.org"))
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager")
install.packages(c("Matrix", "jsonlite"))
BiocManager::install(c("SingleCellExperiment", "scran", "scry", "scDblFinder",
                       "slingshot", "tradeSeq", "DelayedMatrixStats"), ask=FALSE, update=FALSE)
if ("--extended" %in% args) {
    install.packages(c("SoupX", "sctransform", "glmpca", "remotes"))
    remotes::install_github("saeyslab/nichenetr", upgrade="never")
}
