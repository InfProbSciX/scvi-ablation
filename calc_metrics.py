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
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood
from gpytorch.priors import NormalPrior
from torch.distributions import LogNormal

from utils.metrics import calc_bio_metrics, calc_batch_metrics
from model import ScalyEncoder, NNEncoder, NBLikelihood, VariationalLatentVariable
from model import ccScalyEncoder, ccNNEncoder, ccVariationalLatentVariable, LinearEncoder

import argparse

def main(args):
  print(args)
  # plt.ion(); 
  plt.style.use('seaborn-pastel')
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model_dir = f'{args.model_dir}/{args.data}/seed{args.seed}'
  if(args.theta_val == 1):
    model_name = f'{args.model}_{args.preprocessing}_{args.encoder}_{args.kernel}_{args.likelihood}'
  else:
    model_name = f'{args.model}_{args.preprocessing}_{args.encoder}_{args.kernel}_{args.likelihood}_theta{args.theta_val}'
  
  print(model_name)
  
  if('noscale' in args.likelihood):
    if(args.preprocessing != 'libnorm'):
      raise ValueError('Please have libnorm preprocessing when using nblikelihood with noscale.')

  
  # load in data
  print(f'Loading in data {args.data}')
  if('covid_data' in args.data):
    data_dir = "data/COVID_Stephenson/"
    adata = sc.read_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")
    X_covars_keys = ['sample_id']
		# load in data
    Y_rawcounts, X_covars = setup_from_anndata(adata, 
                                 					layer='counts',
                                 					categorical_covariate_keys=X_covars_keys,
                                 					continuous_covariate_keys=None,
                                 					scale_gex=False)
  elif(args.data  == 'splatter_nb' or args.data  == 'test_splatter_nb'):
    data_dir = 'data/simulated_data/'
    adata = sc.read_h5ad(data_dir + "balanced_small.h5ad")
    X_covars_keys = ['sample_id']
    Y_rawcounts, X_covars = setup_from_anndata(adata, 
                                 					layer='counts',
                                 					categorical_covariate_keys=X_covars_keys,
                                 					continuous_covariate_keys=None,
                                 					scale_gex=False)
    Y_rawcounts = Y_rawcounts.to(torch.float32)
    X_covars = X_covars.to(torch.float32)
  elif(args.data == 'splatter_nb_large' or args.data  == 'test_splatter_nb_large'):
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
  elif(args.data == 'splatter_nb_nodropout_large'):
    data_dir = 'data/simulated_data/'
    adata = sc.read_h5ad(data_dir + "nodropout_large.h5ad")
    X_covars_keys = ['sample_id']
    Y_rawcounts, X_covars = setup_from_anndata(adata, 
                                 					layer='counts',
                                 					categorical_covariate_keys=X_covars_keys,
                                 					continuous_covariate_keys=None,
                                 					scale_gex=False)
    Y_rawcounts = Y_rawcounts.to(torch.float32)
    X_covars = X_covars.to(torch.float32)
  elif(args.data == 'test_gaussian'):
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
  
  # set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed) 
  
  # if model is pca
  if(args.model == 'pca'):
    model_name = 'pca'
    ## Calculating Metrics ##
    if('covid_data' in args.data):
      batch_metrics = calc_batch_metrics(adata, embed_key = 'X_pca', batch_key = 'Site', metrics_list = args.batch_metrics)
      torch.save(batch_metrics, f'{model_dir}/{model_name}_batch_metrics_by_Site.pt')
    
    bio_metrics = calc_bio_metrics(adata, embed_key = 'X_pca', batch_key = 'sample_id', metrics_list = args.bio_metrics)
    torch.save(bio_metrics, f'{model_dir}/{model_name}_bio_metrics_by_sampleid.pt')
    
    batch_metrics = calc_batch_metrics(adata, embed_key = 'X_pca', batch_key = 'sample_id', metrics_list = args.batch_metrics)
    torch.save(batch_metrics, f'{model_dir}/{model_name}_batch_metrics_by_sampleid.pt')
  
    ## Generating UMAP ##
    print('\nGenerating UMAP image...')
    umap_seed = 1
    torch.manual_seed(umap_seed)
    np.random.seed(umap_seed)
    random.seed(umap_seed) 
    
    sc.pp.neighbors(adata, n_neighbors=50, use_rep='X_pca', key_added=f'X_{model_name}_k50')
    sc.tl.umap(adata, neighbors_key=f'X_{model_name}_k50')
    adata.obsm[f'umap_{args.data}_{model_name}_seed{args.seed}'] = adata.obsm['X_umap'].copy()
    
    if('covid_data' in args.data):
      plt.rcParams['figure.figsize'] = [10,10]
      col_obs = ['harmonized_celltype', 'Site']
      sc.pl.embedding(adata, f'umap_{args.data}_{model_name}_seed{args.seed}', color = col_obs, legend_loc='on data', size=5,
                  save='_Site.png')
  
    plt.rcParams['figure.figsize'] = [10,10]
    col_obs = ['harmonized_celltype', 'sample_id']
    sc.pl.embedding(adata, f'umap_{args.data}_{model_name}_seed{args.seed}', color = col_obs, legend_loc='on data', size=5,
                  save='_sampleid.png')
    print("Done.")
    return
    
    
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
      linearscvi = scvi.model.LinearSCVI.load(f'{model_dir}/linearNBscVI', adata_ref)
      
      adata_ref.obsm["X_scVI"] = linearscvi.get_latent_representation()
      adata.obsm["X_scVI"] = linearscvi.get_latent_representation(adata_ref)  
    else: 
      raise ValueError(f'Invalid input argument: {args.model} is not a valid scvi input.')
    
    ## Calculating Metrics ##
    if('covid_data' in args.data):
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

    # if('covid_data' in args.data):
    #   plt.rcParams['figure.figsize'] = [10,10]
    #   col_obs = ['harmonized_celltype', 'Site']
    #   sc.pl.embedding(adata, f'umap_{args.data}_{model_name}_seed{args.seed}', color = col_obs, legend_loc='on data', size=5,
    #               save='_Site.png')
  
    # plt.rcParams['figure.figsize'] = [10,10]
    # col_obs = ['harmonized_celltype', 'sample_id']
    # sc.pl.embedding(adata, f'umap_{args.data}_{model_name}_seed{args.seed}', color = col_obs, legend_loc='on data', size=5,
    #               save='_sampleid.png')
    print('Done.')  
    return
  
  # data preprocessing
  print('Starting Data Preprocessing:')
  Y_temp = Y_rawcounts
  if('zeroone' in args.preprocessing):
    print('\tZero-Oneing the data...')
    Y_temp = torch.where(Y_temp > 0, torch.tensor(1), Y_temp)
    
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
  if(args.data == 'covid_data_X'):
    print('Using already built in preprocessed data...')
    Y = torch.tensor(adata.X.todense())
  if(args.data == 'covid_data_X' and args.preprocessing == 'columnstd'):
    Y_temp = torch.tensor(adata.X.todense())
    Y_temp = Y_temp / torch.std(Y_temp, axis = 0)
    Y = Y_temp
  print('Done with data preprocessing!\n')

  q = 10
  (n, d)= Y.shape

  if('learnscale' in args.likelihood):
    learn_scale = True
  else:
    learn_scale = False
    
  if('learntheta' in args.likelihood):
    learn_theta = True
  else:
    learn_theta = False
    
  if('periodic' in args.kernel):
    if('covid_data' in args.data):
      cc_init = torch.tensor(adata.obs['cell_cycle_init'].values.reshape([n,1])).float()
    else:
      raise ValueError(f'Periodic kernel with cell_cycle info only with covid_data right now.')    

  ## Declare encoder ##
  if(args.encoder == 'point'):
    X_latent_init = torch.tensor(adata.obsm['X_pca'][:, 0:q]) # check if this is what we want it to be for covid data
    if('periodic' in args.kernel):
      X_latent = PointLatentVariable(torch.cat([cc_init, X_latent_init], axis =1))
    else:                     
      X_latent = PointLatentVariable(X_latent_init)
  elif(args.encoder == 'vpoint'):
    X_latent_init = torch.tensor(adata.obsm['X_pca'][:, 0:q])
    if('periodic' in args.kernel):
      X_latent = ccVariationalLatentVariable(X_latent_init, Y.shape[1], cc_init = cc_init)
    else:
      X_latent = VariationalLatentVariable(X_latent_init, Y.shape[1])
  elif(args.encoder == 'nnenc'): 
    if('periodic' in args.kernel):
       X_latent = ccNNEncoder(n, q, d, (128, 128), cc_init = cc_init)
    else:
      X_latent = NNEncoder(n, q, d, (128, 128))
  elif(args.encoder == 'scaly'):
    if('periodic' in args.kernel):
      X_latent = ccScalyEncoder(q, d + X_covars.shape[1], learn_scale = learn_scale, Y= Y, cc_init = cc_init)
    else:
      X_latent = ScalyEncoder(q, d + X_covars.shape[1], learn_scale = learn_scale, Y= Y)
  elif(args.encoder == 'scalynocovars'): #('scalynocovars')
    if('periodic' in args.kernel):
      X_latent = ccScalyEncoder(q, d , learn_scale = learn_scale, Y= Y, cc_init = cc_init)
    else:
      X_latent = ScalyEncoder(q, d , learn_scale = learn_scale, Y= Y)
  elif(args.encoder == 'linear1layernocovars'):
    X_latent = LinearEncoder(q, d , Y= Y)
  elif(args.encoder == 'linear1layer'):
    X_latent = LinearEncoder(q, d + X_covars.shape[1], Y= Y)
  else:
    raise ValueError(f"Invalid input argument: {args.encoder}")
  print(f'Using encoder:\n{X_latent}\n')
  
  
  ## Declare GPLVM model ##
  if('periodic' in args.kernel): # add periodic kernel
    period_scale = np.pi 
    pseudotime_dim = True
  else:
    period_scale = np.Inf # no cc/pseudotime tracking
    pseudotime_dim = False
  
  if('rbf' in args.kernel): # use fixed inducing locations for rbf
    learn_inducing_locations = False
  else:
    learn_inducing_locations = True
  
  if(args.kernel == 'linear_' or args.kernel == 'rbf_' or args.kernel == 'linear_linearmean'):
    covariate_dim = 0
    n_inducing = q + 1
  else:
    covariate_dim = len(X_covars.T)
    n_inducing = q + len(X_covars.T) +1
    
  gplvm = GPLVM(n, d, q,
              covariate_dim=covariate_dim,
              n_inducing=n_inducing,
              period_scale=period_scale,
              X_latent=X_latent,
              X_covars=X_covars,
              pseudotime_dim=pseudotime_dim,
              learn_inducing_locations = learn_inducing_locations
             )
  # gplvm.intercept = gpytorch.means.ConstantMean()
  if('linear_' in args.kernel):
    gplvm.random_effect_mean = gpytorch.means.ZeroMean()
    gplvm.covar_module = gpytorch.kernels.LinearKernel(n_inducing - 1) # q + len(X_covars.T)
    
  if(args.kernel == 'linear_linear_linearmean'):
    gplvm.intercept = gpytorch.means.LinearMean(q+len(X_covars.T)) # the dimension of the latent space
    # gplvm.random_effect_mean = gpytorch.means.LinearMean(len(X_covars.T))
    gplvm.covar_module = gpytorch.kernels.LinearKernel(q + len(X_covars.T))
  
  if(args.kernel == 'linear_linearmean'):
    gplvm.intercept = gpytorch.means.LinearMean(q)
    gplvm.covar_module = gpytorch.kernels.LinearKernel(q)
  
  if(args.kernel == 'rbf_rbf'):
    gplvm.intercept = gpytorch.means.ConstantMean()
    gplvm.random_effect_mean = gpytorch.means.ConstantMean()
    gplvm.latent_var_dims = np.arange(0, q)
    latent_covarariance = gpytorch.kernels.RBFKernel(
                    ard_num_dims=len(gplvm.latent_var_dims),
                    active_dims=gplvm.latent_var_dims
                  )
    max_dim = max(gplvm.latent_var_dims, default=-1)
    # max_dim = max(max_dim, max(self.pseudotime_dims, default=-1))
    gplvm.known_var_dims = np.arange(covariate_dim + max_dim, max_dim, -1)
    gplvm.known_var_dims.sort()
    random_effect_covariance = gpytorch.kernels.RBFKernel(
                ard_num_dims=len(gplvm.known_var_dims),
                active_dims=gplvm.known_var_dims
            )
    gplvm.covar_module = latent_covarariance * random_effect_covariance
    
  print(f'Using GPLVM:\n{gplvm}\n')
  
  ## Declare Likelihood ##
  if(args.likelihood == 'gaussianlikelihood'):
    likelihood = GaussianLikelihood(batch_shape=gplvm.batch_shape)
  elif(args.likelihood == 'bernoulli'):
    likelihood = BernoulliLikelihood()
  else: # nblikelihood
    likelihood = NBLikelihood(d, learn_scale = learn_scale, learn_theta = learn_theta)
  print(f'Using likelihood:\n\t{likelihood}')
  
  ## Load in saved models ##
  # if(args.theta_val == 1):
  gplvm.load_state_dict(torch.load(f'{model_dir}/{model_name}_gplvm_state_dict.pt', map_location = torch.device('cpu')))
  # else:
    # gplvm.load_state_dict(torch.load(f'{model_dir}/{model_name}_theta{args.theta_val}_gplvm_state_dict.pt', map_location = torch.device('cpu')))
  print('Loaded in saved model')
  print(gplvm.eval())

  if(args.encoder == 'point'):
    z_params = gplvm.X_latent.X
  elif(args.encoder == 'vpoint'):
    z_params = gplvm.X_latent.q_mu
  elif(args.encoder == 'scaly' or args.encoder == 'linear1layer'):
    z_params = gplvm.X_latent.z_nnet(torch.cat([Y, X_covars], axis=1))
  elif(args.encoder == 'scalynocovars' or args.encoder == 'linear1layernocovars'):
    z_params = gplvm.X_latent.z_nnet(Y)
  else:
    z_params = X_latent.nnet(Y).tanh()*5
  
  if(args.encoder == 'point' or args.encoder == 'vpoint'):
    X_latent_dims = z_params
  else:
    X_latent_dims = z_params[..., :X_latent.latent_dim]
  # num_mc = 200
  # print(f"Approximating latent dimension means with {num_mc} simuls..")
  # Y_input = Y
  # if(args.encoder == 'scaly'):
  #   Y_input = torch.cat([Y, X_covars], axis=1)

  # if('learnscale' in args.likelihood):
  #   sample_f, sample_s = gplvm.X_latent(Y = Y_input)
  #   X_latent_dims += sample_f * sample_s 
  # else:
  #   sample_x = gplvm.X_latent(Y = Y_input)
  # for _ in range(num_mc-1):
  #     sample_f, sample_s = gplvm.X_latent(Y = Y_input)
  #     X_latent_dims += sample_f * sample_s
  # X_latent_dims /= num_mc
  # print("Done approximating in means.")

  adata.obsm[f'X_{model_name}_latent'] = X_latent_dims.detach().numpy()

  filestuff = f'{model_dir}/{model_name}'
  model_name_stuff = model_name
    
  ## Calculating metrics ##
  if('covid_data' in args.data):
    batch_metrics = calc_batch_metrics(adata, embed_key = f'X_{model_name}_latent', batch_key = 'Site', metrics_list = args.batch_metrics)
    torch.save(batch_metrics, f'{filestuff}_batch_metrics_by_Site.pt')
  
  batch_metrics = calc_batch_metrics(adata, embed_key = f'X_{model_name}_latent', batch_key = 'sample_id', metrics_list = args.batch_metrics)
  torch.save(batch_metrics, f'{filestuff}_batch_metrics_by_sampleid.pt')

  bio_metrics = calc_bio_metrics(adata, embed_key = f"X_{model_name}_latent",batch_key = 'sample_id', metrics_list = args.bio_metrics)
  torch.save(bio_metrics, f'{filestuff}_bio_metrics_by_sampleid_.pt')

  ## Generating UMAP image ##
  print('\nGenerating UMAP image...')
  umap_seed = 1
  torch.manual_seed(umap_seed)
  np.random.seed(umap_seed)
  random.seed(umap_seed) 
  
  sc.pp.neighbors(adata, n_neighbors=50, use_rep=f'X_{model_name}_latent', key_added=f'X_{model_name}_k50')
  sc.tl.umap(adata, neighbors_key=f'X_{model_name}_k50')
  adata.obsm[f'umap_{args.data}_{model_name_stuff}_seed{args.seed}'] = adata.obsm['X_umap'].copy()

  # if('covid_data' in args.data):
  #   plt.rcParams['figure.figsize'] = [10,10]
  #   col_obs = ['harmonized_celltype', 'Site']
  #   sc.pl.embedding(adata, f'umap_{args.data}_{model_name_stuff}_seed{args.seed}', color = col_obs, legend_loc='on data', size=5,
  #                 save='_Site.png')
  
  # plt.rcParams['figure.figsize'] = [10,10]
  # col_obs = ['harmonized_celltype', 'sample_id']
  # sc.pl.embedding(adata, f'umap_{args.data}_{model_name_stuff}_seed{args.seed}', color = col_obs, legend_loc='on data', size=5,
  #                 save='_sampleid.png')
  print('Done.')
  
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics of specified train model on specified data")
    parser.add_argument('-m', '--model', type=str, help='overarching model', 
    					default = 'gplvm',
              choices = ['scvi', 'linear_scvi', 'gplvm', 'pca'])
    parser.add_argument('-d', '--data', type=str, help='Data your model was trained on', 
              default = 'covid_data',
    					choices = ['covid_data', 'covid_data_X', 'innate_immunity',
                      'splatter_nb', 'splatter_nb_large', 'splatter_nb_nodropout_large',
                      'test_gaussian', 
                      'test_covid_data', 'test_splatter_nb', 'test_splatter_nb_large'])
    parser.add_argument('-p', '--preprocessing', type = str, help='preprocessing of raw counts',
    					default = 'rawcounts',
    					choices = ['rawcounts', 
                        'libnormlogtrans', 
                        'logtranscolumnstd', 
                        'libnormlogtranscolumnstd',
                        'libnorm',
                        'columnstd',
                        'zeroone'])
    parser.add_argument('-e', '--encoder', type = str, help='type of encoder',
    					default = 'scaly',
    					choices = ['point', 'vpoint', 'nnenc', 'scaly', 'scalynocovars',
                        'linear1layer', 'linear1layernocovars']) 
    parser.add_argument('-k', '--kernel', type = str, help = 'type of kernel',
    					default = 'linear_linear',
    					choices = ['linear_linear', 
                        'periodic_linear', 
                        'rbf_linear', 
                        'rbf_linear_linearmean',
                        'linear_linearmean',
                        'linear_linear_linearmean',
                        'rbf_rbf',
                        'periodicrbf_linear',
                        'linear_',
                        'rbf_'])
    parser.add_argument('-l', '--likelihood', type = str, help = 'likelihood used',
    					default = 'nblikelihoodlearnscalelearntheta',
    					choices = ['gaussianlikelihood', 
                        'nblikelihoodnoscalelearntheta', 
                        'nblikelihoodnoscalefixedtheta1',
                        'nblikelihoodlearnscalelearntheta', 
                        'nblikelihoodlearnscalefixedtheta1',
                        'bernoulli'])
    parser.add_argument('-s', '--seed', type = int, help = 'random seed to initialize everything',
                        default = 42)
    parser.add_argument('--model_dir', type = str, help = 'Directory where all models are stored', 
                        default = 'models')

    parser.add_argument('--bio_metrics', type = str, nargs='*',help = 'List of bio metrics to calculate', default = ['nmi', 'ari', 'cellASW'])
    parser.add_argument('--batch_metrics', type = str,nargs='*', help = 'List of bio metrics to calculate', default = ['batchASW', 'graph_connectivity'])
    parser.add_argument('--cluster_methods', type=str, nargs='+',help = 'List of cluster methods', default = ['kmeans', 'leiden'])
    parser.add_argument('--theta_val', type = float,
                        default = 1)
    
    args = parser.parse_args()

    main(args)
    