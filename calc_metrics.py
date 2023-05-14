import torch
import numpy as np
import random
import scanpy as sc
import argparse

import matplotlib.pyplot as plt
import wandb

import scvi
import os

import gpytorch
from tqdm import trange
from model import GPLVM, LatentVariable, VariationalELBO, BatchIdx, _KL, PointLatentVariable
from utils.preprocessing import setup_from_anndata
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import NormalPrior
from torch.distributions import LogNormal

from utils.metrics import calc_bio_metrics, calc_batch_metrics
from model import ScalyEncoder, NNEncoder, NBLikelihood 

import argparse

def main(args):
  print(args)
  plt.ion(); plt.style.use('seaborn-pastel')
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model_dir = f'{args.model_dir}/{args.data}/seed{args.seed}'
  model_name = f'{args.model}_{args.preprocessing}_{args.encoder}_{args.kernel}_{args.likelihood}'
  
  print(model_name)
  
  # load in data
  print(f'Loading in data {args.data}')
  if(args.data == 'covid_data'):
    data_dir = "data/COVID_Stephenson/"
    adata = sc.read_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")
    X_covars_keys = ['sample_id']
		# load in data
    Y_rawcounts, X_covars = setup_from_anndata(adata, 
                                 					layer='counts',
                                 					categorical_covariate_keys=X_covars_keys,
                                 					continuous_covariate_keys=None,
                                 					scale_gex=False)
  elif(args.data == 'splatter_nb'):
    data_dir = 'data/simulated_data/'
    adata = sc.read_h5ad(data_dir + "balanced3kcells8kgenes.h5ad")
    X_covars_keys = ['sample_id']
    Y_rawcounts, X_covars = setup_from_anndata(adata, 
                                 					layer='counts',
                                 					categorical_covariate_keys=X_covars_keys,
                                 					continuous_covariate_keys=None,
                                 					scale_gex=False)
    Y_rawcounts = Y_rawcounts.to(torch.float32)
    X_covars = X_covars.to(torch.float32)
  elif(args.data == 'splatter_nb_large'):
    data_dir = 'data/simulated_data/'
    adata = sc.read_h5ad(data_dir + "balanced_large.h5ad")
    X_covars_keys = ['sample_id']
    Y_rawcounts, X_covars = setup_from_anndata(adata, 
                                 					layer='counts',
                                 					categorical_covariate_keys=X_covars_keys,
                                 					continuous_covariate_keys=None,
                                 					scale_gex=False)
    Y_rawcounts = Y_rawcounts.to(torch.float32)
    X_covars = X_covars.to(torch.float32)
  else: #innate_immunity
    pass #TODO: add in the data loading here
  print(f'Done loading in  {args.data}\n')
  
  # TODO: add functionality for scvi and linearscvi
  
  # set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed) 
  
  # if model is scvi
  if('scvi' in args.model):
    adata_ref = adata.copy()
    if(args.model == 'scvi'):
      model_name = "nonlinearNBscVI"
      scvi.model.SCVI.setup_anndata(adata_ref, batch_key='sample_id', layer='counts')      
      scvi_ref = scvi.model.SCVI.load(f'{model_dir}/nonlinearNBscVI', adata_ref)
      
      adata_ref.obsm["X_scVI"] = scvi_ref.get_latent_representation()
      adata.obsm["X_scVI"] = scvi_ref.get_latent_representation(adata_ref)
    elif(args.model == 'linear_scvi'):
      model_name = "linearNBscVI"
      scvi.model.LinearSCVI.setup_anndata(adata_ref, batch_key='sample_id', layer='counts')
      linearscvi = scvi.model.LinearSCVI.load('models/linearNBscVI', adata_ref)
      
      adata_ref.obsm["X_scVI"] = linearscvi.get_latent_representation()
      adata.obsm["X_scVI"] = linearscvi.get_latent_representation(adata_ref)  
    else: 
      raise ValueError(f'Invalid input argument: {args.model} is not a valid scvi input.')
    
    ## Calculating Metrics ##
    if(args.data == 'covid_data'):
      batch_metrics = calc_batch_metrics(adata, embed_key = 'X_scVI', batch_key = 'Site', metrics_list = args.batch_metrics)
      torch.save(batch_metrics, f'{model_dir}/{model_name}_batch_metrics_by_Site.pt')
    
    bio_metrics = calc_bio_metrics(adata, embed_key = 'X_scVI', batch_key = 'sample_id', metrics_list = args.bio_metrics)
    torch.save(bio_metrics, f'{model_dir}/{model_name}_bio_metrics_by_sampleid.pt')
    
    batch_metrics = calc_batch_metrics(adata, embed_key = 'X_scVI', batch_key = 'sample_id', metrics_list = args.batch_metrics)
    torch.save(batch_metrics, f'{model_dir}/{model_name}_batch_metrics_by_sampleid.pt')
  
    ## Generating UMAP ##
    print('\nGenerating UMAP image...')
    umap_seed = 1
    torch.manual_seed(umap_seed)
    np.random.seed(umap_seed)
    random.seed(umap_seed) 
    
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=f'X_scVI', key_added=f'X_{model_name}_k50')
    sc.tl.umap(adata, neighbors_key=f'X_{model_name}_k50')
    adata.obsm[f'umap_{args.data}_{model_name}_seed{args.seed}'] = adata.obsm['X_umap'].copy()

    if(args.data == 'covid_data'):
      plt.rcParams['figure.figsize'] = [10,10]
      col_obs = ['harmonized_celltype', 'Site']
      sc.pl.embedding(adata, f'umap_{args.data}_{model_name}_seed{args.seed}', color = col_obs, legend_loc='on data', size=5,
                  save='_Site.png')
  
    plt.rcParams['figure.figsize'] = [10,10]
    col_obs = ['harmonized_celltype', 'sample_id']
    sc.pl.embedding(adata, f'umap_{args.data}_{model_name}_seed{args.seed}', color = col_obs, legend_loc='on data', size=5,
                  save='_sampleid.png')
    print('Done.')  
    return
  
  
  # data preprocessing
  print('Starting Data Preprocessing:')
  Y_temp = Y_rawcounts
  if('libnorm' in args.preprocessing):
    print('\tLibrary normalizing...')
    Y_temp = Y_temp/Y_temp.sum(1, keepdim = True) * 10000
  if('logtrans' in args.preprocessing):
    print('\tLog transforming...')
    Y_temp = torch.log(Y_temp + 1)
  if('columnstd' in args.preprocessing):
    print('\tColumn standardizing...')
    Y_temp = Y_temp / torch.std(Y_temp, axis = 0)
  Y = Y_temp
  print('Done with data preprocessing!\n')

  q = 10
  (n, d)= Y.shape
  period_scale = np.Inf # no period/cell-cycle shenanigans

  if('learnscale' in args.likelihood):
    learn_scale = True
  else:
    learn_scale = False
  if('learntheta' in args.likelihood):
    learn_theta = True
  else:
    learn_theta = False

  ## Declare encoder ##
  if(args.encoder == 'point'):
    X_latent = PointLatentVariable(torch.randn(n, q))
  elif(args.encoder == 'nnenc'): 
    X_latent = NNEncoder(n, q, d, (128, 128))
  elif(args.encoder == 'scaly'):
    X_latent = ScalyEncoder(q, d + X_covars.shape[1], learn_scale = learn_scale, Y= Y)
  else: #('scalynocovars')
    X_latent = ScalyEncoder(q, d , learn_scale = learn_scale, Y= Y)
  print(f'Using encoder:\n{X_latent}\n')
  
  ## Declare GPLVM model ##
  if('periodic' in args.kernel):
    period_scale = np.pi # stand-in rn
  else:
    period_scale = np.Inf # no cc/pseudotime tracking
  gplvm = GPLVM(n, d, q,
              covariate_dim=len(X_covars.T),
              n_inducing=q + len(X_covars.T)+1,
              period_scale=period_scale,
              X_latent=X_latent,
              X_covars=X_covars,
              pseudotime_dim=False
             )
  gplvm.intercept = gpytorch.means.ConstantMean()
  gplvm.random_effect_mean = gpytorch.means.ZeroMean()
  if('linear_' in args.kernel):
    gplvm.covar_module = gpytorch.kernels.LinearKernel(q + len(X_covars.T))
  print(f'Using GPLVM:\n{gplvm}\n')
  
  ## Declare Likelihood ##
  if(args.likelihood == 'gaussianlikelihood'):
    likelihood = GaussianLikelihood(batch_shape=gplvm.batch_shape)
  else: # nblikelihood
    likelihood = NBLikelihood(d, learn_scale = learn_scale, learn_theta = learn_theta)
  print(f'Using likelihood:\n\t{likelihood}')
  
  ## Load in saved models ##
  gplvm.load_state_dict(torch.load(f'{model_dir}/{model_name}_gplvm_state_dict.pt', map_location = torch.device('cpu')))
  print('Loaded in saved model')
  print(gplvm.eval())

  if(args.encoder == 'point'):
    z_params = gplvm.X_latent.X
  elif(args.encoder == 'scaly'):
    z_params = gplvm.X_latent.z_nnet(torch.cat([Y, X_covars], axis=1))
  elif(args.encoder == 'scalynocovars'):
    z_params = gplvm.X_latent.z_nnet(Y)
  else:
    z_params = X_latent.nnet(Y)
  
  if(args.encoder == 'point'):
    X_latent_dims = z_params
  else:
    X_latent_dims = z_params[..., :X_latent.latent_dim]
    
  
  adata.obsm[f'X_{model_name}_latent'] = X_latent_dims.detach().numpy()

  ## Calculating metrics ##
  if(args.data == 'covid_data'):
    batch_metrics = calc_batch_metrics(adata, embed_key = f'X_{model_name}_latent', batch_key = 'Site', metrics_list = args.batch_metrics)
    torch.save(batch_metrics, f'{model_dir}/{model_name}_batch_metrics_by_Site.pt')
  
  batch_metrics = calc_batch_metrics(adata, embed_key = f'X_{model_name}_latent', batch_key = 'sample_id', metrics_list = args.batch_metrics)
  torch.save(batch_metrics, f'{model_dir}/{model_name}_batch_metrics_by_sampleid.pt')

  bio_metrics = calc_bio_metrics(adata, embed_key = f"X_{model_name}_latent",batch_key = 'sample_id', metrics_list = args.bio_metrics)
  torch.save(bio_metrics, f'{model_dir}/{model_name}_bio_metrics_by_sampleid_.pt')

  ## Generating UMAP image ##
  print('\nGenerating UMAP image...')
  umap_seed = 1
  torch.manual_seed(umap_seed)
  np.random.seed(umap_seed)
  random.seed(umap_seed) 
  
  sc.pp.neighbors(adata, n_neighbors=50, use_rep=f'X_{model_name}_latent', key_added=f'X_{model_name}_k50')
  sc.tl.umap(adata, neighbors_key=f'X_{model_name}_k50')
  adata.obsm[f'umap_{args.data}_{model_name}_seed{args.seed}'] = adata.obsm['X_umap'].copy()

  if(args.data == 'covid_data'):
    plt.rcParams['figure.figsize'] = [10,10]
    col_obs = ['harmonized_celltype', 'Site']
    sc.pl.embedding(adata, f'umap_{args.data}_{model_name}_seed{args.seed}', color = col_obs, legend_loc='on data', size=5,
                  save='_Site.png')
  
  plt.rcParams['figure.figsize'] = [10,10]
  col_obs = ['harmonized_celltype', 'sample_id']
  sc.pl.embedding(adata, f'umap_{args.data}_{model_name}_seed{args.seed}', color = col_obs, legend_loc='on data', size=5,
                  save='_sampleid.png')
  print('Done.')
  
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics of specified train model on specified data")
    parser.add_argument('-m', '--model', type=str, help='overarching model', 
    					default = 'gplvm',
              choices = ['scvi', 'linear_scvi', 'gplvm'])
    parser.add_argument('-d', '--data', type=str, help='Data your model was trained on', 
              default = 'covid_data',
    					choices = ['covid_data', 'innate_immunity', 'splatter_nb', 'splatter_nb_large'])
    parser.add_argument('-p', '--preprocessing', type = str, help='preprocessing of raw counts',
    					default = 'rawcounts',
    					choices = ['rawcounts', 
                        'libnormlogtrans', 
                        'logtranscolumnstd', 
                        'libnormlogtranscolumnstd'])
    parser.add_argument('-e', '--encoder', type = str, help='type of encoder',
    					default = 'point',
    					choices = ['point', 'nnenc', 'scaly', 'scalynocovars']) 
    parser.add_argument('-k', '--kernel', type = str, help = 'type of kernel',
    					default = 'linear_linear',
    					choices = ['linear_linear', 
                        'periodic_linear', 
                        'rbf_linear', 
                        'periodicrbf_linear'])
    parser.add_argument('-l', '--likelihood', type = str, help = 'likelihood used',
    					default = 'nblikelihoodnoscalelearntheta',
    					choices = ['gaussianlikelihood', 
                        'nblikelihoodnoscalelearntheta', 
                        'nblikelihoodnoscalefixedtheta1',
                        'nblikelihoodlearnscalelearntheta', 
                        'nblikelihoodlearnscalefixedtheta1'])
    parser.add_argument('-s', '--seed', type = int, help = 'random seed to initialize everything',
                        default = 42)
    parser.add_argument('--model_dir', type = str, help = 'Directory where all models are stored', 
                        default = 'models')

    parser.add_argument('--bio_metrics', type = str, nargs='*',help = 'List of bio metrics to calculate', default = ['nmi', 'ari', 'cellASW'])
    parser.add_argument('--batch_metrics', type = str,nargs='*', help = 'List of bio metrics to calculate', default = ['batchASW', 'graph_connectivity'])
    parser.add_argument('--cluster_methods', type=str, nargs='+',help = 'List of cluster methods', default = ['kmeans', 'leiden'])
    
    args = parser.parse_args()

    main(args)
    