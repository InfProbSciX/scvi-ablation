# run this from main folder (not utils)

import pandas as pd
import scanpy as sc

data_dir = 'data/simulated_data'

count = pd.read_table(f'{data_dir}/counts20k10k_seed1.txt')
cellinfo = pd.read_table(f'{data_dir}/cellinfo20k10k_seed1.txt')
geneinfo = pd.read_table(f'{data_dir}/geneinfo20k10k_seed1.txt')
adata = sc.AnnData(count.values)
adata.obs = cellinfo[['Batch', 'Group', 'ExpLibSize']].rename(columns={'Batch': 'sample_id', 'Group': 'harmonized_celltype'})
adata.var = geneinfo
adata.var_names.name = 'Gene'
adata.obs_names.name = 'CellID'
adata.layers['counts'] = adata.X.copy()
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.write_h5ad(f"{data_dir}/balanced_large.h5ad")


