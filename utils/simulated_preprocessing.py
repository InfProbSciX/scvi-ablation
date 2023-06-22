# run this from main folder (not utils)

import pandas as pd
import scanpy as sc
import numpy as np

data_dir = 'data/simulated_data'
datasets = ['_large_nodropout_seed1'] #'20k10k_seed1', '_seed1']
names = ['nodropout_large'] #'balanced_large', 'balanced_small']

for i in range(len(datasets)):
  # import ipdb; ipdb.set_trace()
  count = pd.read_table(f'{data_dir}/counts{datasets[i]}.txt')
  cellinfo = pd.read_table(f'{data_dir}/cellinfo{datasets[i]}.txt')
  geneinfo = pd.read_table(f'{data_dir}/geneinfo{datasets[i]}.txt')
  adata = sc.AnnData(count.values)
  adata.obs = cellinfo[['Batch', 'Group', 'ExpLibSize']].rename(columns={'Batch': 'sample_id', 'Group': 'harmonized_celltype'})
  adata.var = geneinfo
  adata.var_names.name = 'Gene'
  adata.obs_names.name = 'CellID'
  adata.layers['counts'] = adata.X.copy()
  sc.pp.filter_cells(adata, min_genes=200)
  sc.pp.filter_genes(adata, min_cells=3)
  
  adata_pp = adata.copy()
  
  sc.pp.log1p(adata_pp)
  sc.pp.scale(adata_pp)
  sc.tl.pca(adata_pp)
  
  sc.pp.neighbors(adata_pp, n_neighbors=50, n_pcs=10)
  sc.tl.umap(adata_pp)
  
  adata.obsm['X_pca'] = adata_pp.obsm['X_pca'].copy()
  adata.obsm['X_umap_pca'] = adata_pp.obsm['X_umap'].copy()
  
  adata.obsp['pca_distances'] = adata_pp.obsp['distances'].copy()
  adata.obsp['pca_connectivities'] = adata_pp.obsp['connectivities'].copy()
  
  adata.write_h5ad(f"{data_dir}/{names[i]}.h5ad")


