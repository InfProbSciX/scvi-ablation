import torch
import numpy as np
import random
import scanpy as sc
import argparse

import matplotlib.pyplot as plt
# import wandb

import scvi
import os, sys

import gpytorch
from tqdm import trange
from model import GPLVM, LatentVariable, VariationalELBO, BatchIdx, _KL, PointLatentVariable
from utils.preprocessing import setup_from_anndata
from gpytorch.likelihoods import GaussianLikelihood, BernoulliLikelihood

from model import ScalyEncoder, NNEncoder, NBLikelihood, VariationalLatentVariable, LinearEncoder
from model import ccScalyEncoder, ccNNEncoder, ccVariationalLatentVariable #, NBLikelihoodwExp

import argparse

## define training functions
def train(gplvm, likelihood, Y, learn_scale, args,
          seed, epochs=15, batch_size=300, lr=0.005):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    n = len(Y)
    steps = int(np.ceil(epochs*n/batch_size))
    elbo_func = VariationalELBO(likelihood, gplvm, num_data=n)
    optimizer = torch.optim.Adam([
        dict(params=gplvm.parameters(), lr=lr),
        dict(params=likelihood.parameters(), lr=lr)
    ])

    losses = []; idx = BatchIdx(n, batch_size).idx()
    iterator = trange(steps, leave=False)
    for i in iterator:
        batch_index = next(idx)
        optimizer.zero_grad()
        
        Y_batch = Y[batch_index]
        try:
          if(learn_scale):
            if(args.encoder == 'scaly'):
              X_l, S_l = gplvm.X_latent(Y = Y_batch, X_covars = gplvm.X_covars[batch_index], batch_index = batch_index)
            elif(args.encoder == 'scalynocovars'):
              X_l, S_l = gplvm.X_latent(Y = Y_batch, X_covars = None, batch_index = batch_index)
            else:
              raise ValueError(f'Invalid input argument: {args.encoder} given learn_scale. Please choose a variation of scalyencoder.')
          else:
            if(args.encoder == 'point' or args.encoder == 'vpoint'):
              X_l = gplvm.X_latent(batch_index = batch_index)
            elif(args.encoder == 'nnenc'):
              X_l = gplvm.X_latent(Y = Y_batch, batch_index = batch_index)
            elif(args.encoder == 'scalynocovars'):
              X_l = gplvm.X_latent(Y = Y_batch, X_covars = None, batch_index = batch_index)
            elif(args.encoder == 'scaly'):
              X_l = gplvm.X_latent(Y = Y_batch, X_covars = gplvm.X_covars[batch_index], batch_index = batch_index) 
            elif(args.encoder == 'linear1layernocovars'):
              X_l = gplvm.X_latent(Y = Y_batch, X_covars = None, batch_index = batch_index)
            elif(args.encoder == 'linear1layer'):
              X_l = gplvm.X_latent(Y = Y_batch, X_covars = gplvm.X_covars[batch_index], batch_index = batch_index) 
            else:
              raise ValueError(f"Invalid input argument: {args.encoder}")
          if('_linear' in args.kernel or  '_rbf' in args.kernel):
            if(args.kernel == 'linear_linearmean'):
              X_sample = X_l
            else:
              X_sample = torch.cat((X_l, gplvm.X_covars[batch_index]), axis=1)
          else: 
            X_sample = X_l
          gplvm_dist = gplvm(X_sample)
          if(learn_scale):
              loss = -elbo_func(gplvm_dist, Y_batch.T, scale=S_l).sum()
          else:
              loss = -elbo_func(gplvm_dist, Y_batch.T).sum()                 # use this when scaling factor is not learned
          # if(loss > 350e5):
          #   import ipdb; ipdb.set_trace()
        except:
          import ipdb; ipdb.set_trace()
          model_dir = f'{args.model_dir}/{args.data}/seed{args.seed}'
          model_name = f'{args.model}_{args.preprocessing}_{args.encoder}_{args.kernel}_{args.likelihood}'
          if(args.theta_val == 1):
            filestuff = f'{model_dir}/{model_name}'
          else:
            filestuff = f'{model_dir}/{model_name}_theta{args.theta_val}' 
          torch.save(losses, f'{filestuff}_losses_error.pt')
          torch.save(likelihood.state_dict(), f'{filestuff}_likelihood_state_dict_error.pt')
          torch.save(gplvm.state_dict(), f'{filestuff}_gplvm_state_dict_error.pt')
          plt.plot(losses)
          plt.savefig(f"{filestuff}_losses_error.png")
        
        losses.append(loss.item())
        # wandb.log({'train loss': loss.item()})
        iter_descrip = f'L:{np.round(loss.item(), 2)}'

        iterator.set_description(iter_descrip)
        loss.backward()
        optimizer.step()
        if('periodic' in args.likelihood):
          gplvm.X_latent.cc_latent.data.clamp_(0., 2*np.pi) # keep cell-cycle between 0 and 2pi

    return losses

def main(args):
  print(args)
  # plt.ion();
  plt.style.use('seaborn-pastel')
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model_dir = f'{args.model_dir}/{args.data}/seed{args.seed}'
  model_name = f'{args.model}_{args.preprocessing}_{args.encoder}_{args.kernel}_{args.likelihood}'
  
  print(model_name)
  print(f'using {device}')
  
  if not os.path.exists(model_dir):
    print(f'{model_dir} does not exist')
    os.makedirs(model_dir, exist_ok=True)
    print(f'{model_dir} created.')
  
  if('noscale' in args.likelihood):
    if(args.preprocessing != 'libnorm'):
      raise ValueError('Please have libnorm preprocessing when using nblikelihood with noscale.')
  
  
  # load in data
  print(f'Loading in data {args.data}')
  if('covid_data' in args.data):
    data_dir = "data/COVID_Stephenson/"
    if(args.data == 'full_covid_data'):
      adata = sc.read_h5ad(data_dir + 'Stephenson.wcellcycle.h5ad')
      X_covars_keys = ['sample_id']
    else:
      adata = sc.read_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")
      X_covars_keys = ['sample_id']
		# load in data
    Y_rawcounts, X_covars = setup_from_anndata(adata, 
                                 					layer='counts',
                                 					categorical_covariate_keys=X_covars_keys,
                                 					continuous_covariate_keys=None,
                                 					scale_gex=False)
  elif(args.data == 'splatter_nb' or args.data == 'test_splatter_nb'):
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
  elif(args.data == 'splatter_nb_large' or args.data == 'test_splatter_nb_large'):
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
  
  # if scvi:
  if('scvi' in args.model):
    arches_params = dict(
      use_layer_norm="both",
      use_batch_norm="none",
      encode_covariates=True,
      dropout_rate=0.2,
      n_layers=2,
      gene_likelihood = 'nb',
    )

    train_params = dict(
      max_epochs=500, 
      train_size = 1.0,
      batch_size = 300,
      plan_kwargs = {'lr': 0.005}, 
    )
    adata_ref = adata.copy()
    
    if(args.model == 'scvi'):
      scvi.model.SCVI.setup_anndata(adata_ref, batch_key='sample_id', layer='counts')
      scvi_ref = scvi.model.SCVI(adata_ref, **arches_params)
      scvi_ref.train(**train_params)

      scvi_ref.save(f'{model_dir}/nonlinearNBscVI/')
      losses = scvi_ref.history['elbo_train']
      torch.save(losses, f'{model_dir}/nonlinearNBscVI_losses.pt')
      losses.plot()
      plt.savefig(f"{model_dir}/nonlinearNBscVI_losses.png")
    elif(args.model == 'linear_scvi'):
      scvi.model.LinearSCVI.setup_anndata(adata_ref, batch_key='sample_id', layer='counts')
      linearscvi = scvi.model.LinearSCVI(adata_ref, **arches_params)
      linearscvi.train(**train_params)

      linearscvi.save(f'{model_dir}/linearNBscVI/')
      losses = linearscvi.history['elbo_train']
      torch.save(losses, f'{model_dir}/linearNBscVI_losses.pt')
      losses.plot()
      plt.savefig(f"{model_dir}/linearNBscVI_losses.png")
    else: 
      raise ValueError(f'Invalid input argument: {args.model} is not a valid scvi input.')
    sys.exit()
    
  # if gplvm - then continue
  
  # data preprocessing
  print('Starting Data Preprocessing:')
  Y_temp = Y_rawcounts
  if('zeroone' in args.preprocessing):
    print('\tZero-Oneing the data...')
    Y_temp = torch.where(Y_temp > 0, torch.tensor(1), Y_temp)
    
  if('libnorm' in args.preprocessing):
    print('\tLibrary normalizing...')
    # Y_temp = Y_temp/Y_temp.sum(1, keepdim = True) * 10000
    if('noscale' in args.likelihood):
      Y_temp = Y_temp/Y_temp.sum(1, keepdim = True) * 5000
    else:
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
  # if periodic, initialize with cc_init with point latent variable #
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
              pseudotime_dim = pseudotime_dim,
              learn_inducing_locations = learn_inducing_locations
             )
  # gplvm.intercept = gpytorch.means.ConstantMean()
  if('linear_' in args.kernel):
    gplvm.random_effect_mean = gpytorch.means.ZeroMean()
    gplvm.covar_module = gpytorch.kernels.LinearKernel(n_inducing - 1) # q + len(X_covars.T)
  
  if(args.kernel == 'linear_linear_linearmean'):
    gplvm.intercept = gpytorch.means.LinearMean(q + len(X_covars.T)) 
    # gplvm.intercept = gpytorch.means.ConstantMean()
    # gplvm.random_effect_mean = gpytorch.means.LinearMean(len(X_covars.T), bias = False) # gpytorch.means.LinearMean(len(X_covars.T))
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
  elif('wexp' in args.likelihood):
    likelihood = NBLikelihoodwExp(d, learn_scale = learn_scale, learn_theta = learn_theta, theta_val = args.theta_val)
  elif(args.likelihood == 'bernoulli'):
    likelihood = BernoulliLikelihood()
  else: # nblikelihood
    likelihood = NBLikelihood(d, learn_scale = learn_scale, learn_theta = learn_theta, theta_val = args.theta_val)
  print(f'Using likelihood:\n\t{likelihood}')
  
  ## Train the Model ##
  val_split = 0
  
  config = {
    "learning_rate": args.learning_rate,
    "epochs": args.epochs, 
    "batch_size": args.batch_size,
    'likelihood': likelihood,
    'X_latent': gplvm.X_latent,
    'n_inducing': q + len(X_covars.T) + 1,
    'covariate_dim': len(X_covars.T),
    'validation_split': val_split,
    'eval_iter': 100,
    'period_scale': period_scale,
    'X_covars': X_covars,
    'X_covar_keys': X_covars_keys,
    'pseudotime_dim': False,
    'seed': args.seed,
    'elbo_func': f'VariationalELBO(likelihood, gplvm, num_data={n - int(np.floor(n*val_split))}), learning inducing loc',
    'gplvm': gplvm,
    'learn_scale': learn_scale,
    'learn_theta': learn_theta,
    'args': args
  } 
  
  print('Initializing and training model...')
  # wandb.init(project="scvi-ablation", entity="ml-at-cl", name = f'{args.data}_{model_name}', config = config)

  if torch.cuda.is_available():
    Y = Y.cuda()
    gplvm = gplvm.cuda()
    gplvm.X_covars = gplvm.X_covars.cuda()
    likelihood = likelihood.cuda()
    gplvm.X_latent = gplvm.X_latent.cuda()
    
  losses = train(gplvm=gplvm, likelihood=likelihood, Y=Y, args =args,
               seed = config['seed'], learn_scale = config['learn_scale'], 
                lr=config['learning_rate'], epochs=config['epochs'], batch_size=config['batch_size']) 

  if(args.theta_val == 1):
    filestuff = f'{model_dir}/{model_name}'
  else:
    filestuff = f'{model_dir}/{model_name}_theta{args.theta_val}' 
  torch.save(losses, f'{filestuff}_losses.pt')
  torch.save(likelihood.state_dict(), f'{filestuff}_likelihood_state_dict.pt')
  torch.save(gplvm.state_dict(), f'{filestuff}_gplvm_state_dict.pt')
  plt.plot(losses)
  plt.savefig(f"{filestuff}_losses.png")

  # wandb.finish()
  print('Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics of specified train model on specified data")
    parser.add_argument('-m', '--model', type=str, help='overarching model', 
    					default = 'gplvm',
              choices = ['scvi', 'linear_scvi', 'gplvm'])
    parser.add_argument('-d', '--data', type=str, help='Data your model was trained on', 
              default = 'full_covid_data',
    					choices = ['covid_data', 'covid_data_X', 'full_covid_data', 'innate_immunity', 
                        'splatter_nb', 'splatter_nb_large', 'splatter_nb_nodropout_large',
                        'test_gaussian',
                        'test_covid_data', 'test_splatter_nb', 'test_splatter_nb_large'])
    parser.add_argument('-p', '--preprocessing', type = str, help='preprocessing of raw counts',
    					default = 'libnorm',
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
    					default = 'rbf_linear',
    					choices = ['linear_linear', 
                        'periodic_linear', 
                        'rbf_linear', 
                        'rbf_linear_linearmean', # this one doesn't make sense...
                        'linear_linearmean',
                        'linear_linear_linearmean',
                        'periodicrbf_linear',
                        'linear_',
                        'rbf_',
                        'rbf_rbf'])
    parser.add_argument('-l', '--likelihood', type = str, help = 'likelihood used',
    					default = 'nblikelihoodnoscalefixedtheta1',
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
    parser.add_argument('--epochs', type = int, help = 'number of epochs to run',
                        default = 15)
    parser.add_argument('--theta_val', type = float,
                        default = 1000000)
    parser.add_argument('--batch_size', type = int,
                        default = 300)
    parser.add_argument('--learning_rate', type = float,
                        default = 0.005)
    
    args = parser.parse_args()

    main(args)
    
