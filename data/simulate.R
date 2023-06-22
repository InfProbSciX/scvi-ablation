library(splatter)

base_params <- newSplatParams(
                        nGenes = 10000,
                        mean.shape = 0.34,
                        mean.rate = 7.68,
                        lib.loc = 7.64,
                        lib.scale = 0.78,
                        out.prob = 0.00286,
                        out.facLoc = 6.15,
                        out.facScale = 0.49,
                        bcv.common = 0.448,
                        bcv.df = 22.087)

for (seed in 1:1){
    
    sim_params <- setParams(
        base_params,
        batchCells     = c(10000, 10000), #c(2000, 2000),
        batch.facLoc   = c(0.10, 0.11),
        batch.facScale = c(0.50, 0.50),
        # Groups with equal probabilities
        group.prob     = rep(1, 7) / 7,
        # Differential expression by group
        de.prob        = c(0.10, 0.12, 0.08, 0.20, 0.12, 0.10, 0.16),
        de.facLoc      = c(0.10, 0.08, 0.12, 0.18, 0.06, 0.20, 0.14),
        de.facScale    = rep(1.0, 7),
        # Seed
        seed           = seed,
        dropout.type   = 'experiment' # 'none'
    )

    # Simulate the full dataset that we will downsample
    sim <- splatSimulateGroups(sim_params)
    
    counts <- as.data.frame(t(as.array(counts(sim))))
    cellinfo <- as.data.frame(colData(sim))
    geneinfo <- as.data.frame(rowData(sim))
    
    write.table(counts, file ="counts20k10k_seed1.txt", sep = "\t", row.names = TRUE, col.names = TRUE)
    write.table(geneinfo, file = "geneinfo20k10k_seed1.txt", sep = "\t", row.names = TRUE, col.names = TRUE)
    write.table(cellinfo, file = "cellinfo20k10k_seed1.txt", sep = "\t", row.names = TRUE, col.names = TRUE)
}



# # estimate covid data
# install.packages("SeuratDisk")
# library(SeuratDisk)
library(Seurat)
library(SeuratDisk)
h5ad_file <- "Dropbox/BrightHeart/2022-2023/ACS/Research/scvi-ablation/data/COVID_Stephenson/Stephenson.subsample.100k.h5ad"
# covid_data_sce_object <- ReadH5AD(file = h5ad_file)
# print(sce_object)
# 
Convert(h5ad_file, ".h5seurat")
h5Seurat_file <- "Dropbox/BrightHeart/2022-2023/ACS/Research/scvi-ablation/data/COVID_Stephenson/Stephenson.subsample.100k.h5Seurat"
seuratObject <- LoadH5Seurat(h5Seurat_file)

library(anndata)
ad <- read_h5ad(h5ad_file)
splatEstimate(ad$layers['counts'], params = newSplatParams())


