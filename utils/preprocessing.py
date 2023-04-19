
from typing import List, Optional

from anndata import AnnData

import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import os, torch


def setup_from_anndata(
        adata: AnnData,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        layer: Optional[str] = None,
        scale_gex: Optional[bool] = False,
        scale_continuous_covariate: Optional[bool] = False,
    ):
        """
        Parse model inputs from AnnData object

        Parameters
        ----------
        - adata: AnnData object
        - categorical_covariate_keys: keys in adata.obs that correspond to categorical data.
        - continuous_covariate_keys: keys in adata.obs that correspond to continuous data.
        - layer: if not None, uses this as the key in adata.layers for gene expression data (log-normalized).
        - scale_gex: boolean indicating whether gene expression values should be scaled to standard deviation
        - scale_continuous_covariate: boolean indicating whether continuous covariate values should be scaled to standard deviation and mean centered

        Returns:
        --------
        - Y: tensor of gene expression data 
        - X_covars: model matrix of all covariates to model
        """
        if type(adata) != AnnData:
            raise(TypeError("Input is not AnnData object"))

        ## Make model matrix from covariates
        if categorical_covariate_keys is not None:
            cat_obs = adata.obs[categorical_covariate_keys].copy()
            cat_model_mat = pd.get_dummies(
                cat_obs, 
                columns=categorical_covariate_keys
            )
        else:
            cat_model_mat = None

        if continuous_covariate_keys is not None:
            cont_obs = adata.obs[continuous_covariate_keys].copy()
            if scale_continuous_covariate:
                cont_obs -= cont_obs.mean()
                cont_obs /= cont_obs.std()
        else:
            cont_obs = None

        if cat_model_mat is not None or cont_obs is not None:
            model_mat = pd.concat([cont_obs, cat_model_mat], axis=1)
            X_covars = torch.tensor(model_mat.values).float()
        else:
            model_mat = None
            X_covars = None

        ## Get gene expression data
        if layer is None:
            Y_gex = adata.X.copy()
        else:
            Y_gex = adata.layers[layer].copy()

        if sp.issparse(Y_gex):
            Y_gex = Y_gex.todense()

        Y = torch.tensor(Y_gex)

        if scale_gex:
            Y /= Y.std(axis=0)

        ## Store components in adata in place
        if model_mat is not None:
            adata.obsm['model_mat'] = model_mat.copy()
        adata.uns['model_setup_params'] = {
            'categorical_covariate_keys':categorical_covariate_keys,
            'continuous_covariate_keys':continuous_covariate_keys
        }

        return(Y, X_covars)
