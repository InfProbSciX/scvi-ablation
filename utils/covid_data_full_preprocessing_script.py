
import torch
import numpy as np
import random
import scanpy as sc

import matplotlib.pyplot as plt
import wandb
plt.ion(); plt.style.use('seaborn-pastel')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#####################################################################
# Data Prep for covid

import os, scvi

data_dir = "data/COVID_Stephenson/"

if not os.path.exists(os.path.join(data_dir, "Stephenson.wcellcycle.h5ad")):
    os.makedirs(data_dir, exist_ok=True)

    # import gdown
    # gdown.download(id='1Sw5UnLPRLD-4fyFItO4bQbmJioUjABvf', output=data_dir)

    from initialise_latent_var import get_CC_effect_init, cc_genes

    adata_full = sc.read_h5ad(data_dir + "Stephenson.h5ad")
    adata_full.obs = adata_full.obs[['sample_id', 'Site', 'harmonized_celltype']].copy()

    adata_full.obs['cell_cycle_init'] = get_CC_effect_init(adata_full, cc_genes)
    # adata = sc.pp.subsample(adata_full, n_obs=100000, copy=True)

    adata_full.layers['counts'] = adata_full.layers['raw'].copy()
    # # sample most variable genes
    sc.pp.highly_variable_genes(adata_full, n_top_genes=5000, subset=True)
    adata_full.write_h5ad(data_dir + "Stephenson.wcellcycle.h5ad")

# adata = sc.read_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")