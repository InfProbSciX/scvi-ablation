import torch
import numpy as np
import random
import scanpy as sc
import argparse

import matplotlib.pyplot as plt
import wandb

import gpytorch
from tqdm import trange
from model import GPLVM, LatentVariable, VariationalELBO, BatchIdx, _KL, PointLatentVariable
from utils.preprocessing import setup_from_anndata
from gpytorch.likelihoods import GaussianLikelihood

from model import ScalyEncoder, NNEncoder, NBLikelihood 

import argparse

## define training functions
def train(gplvm, likelihood, Y, learn_scale, encoder,
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

        if(learn_scale):
          if(encoder == 'scaly'):
            X_l, S_l = gplvm.X_latent(Y = Y_batch, X_covars = gplvm.X_covars[batch_index])
          elif(encoder == 'scalynocovars'):
            X_l, S_l = gplvm.X_latent(Y = Y_batch, X_covars = None)
          else:
            raise ValueError(f'Invalid input argument: {encoder} given learn_scale. Please choose a variation of scalyencoder.')
        else:
          if(encoder == 'point'):
            X_l = gplvm.X_latent(batch_index = batch_index)
          elif(encoder == 'nnenc'):
            X_l = gplvm.X_latent(Y = Y_batch)
          elif(encoder == 'scalynocovars'):
            X_l = gplvm.X_latent(Y = Y_batch, X_covars = None)
          elif(encoder == 'scaly'):
            X_l = gplvm.X_latent(Y = Y_batch, X_covars = gplvm.X_covars[batch_index]) 
          else:
            raise ValueError(f'Invalid input argument: {encoder}')
        X_sample = torch.cat((X_l, gplvm.X_covars[batch_index]), axis=1)
        gplvm_dist = gplvm(X_sample)
        if(learn_scale):
            loss = -elbo_func(gplvm_dist, Y_batch.T, scale=S_l).sum()
        else:
            loss = -elbo_func(gplvm_dist, Y_batch.T).sum()                 # use this when scaling factor is not learned
        
        losses.append(loss.item())
        wandb.log({'train loss': loss.item()})
        iter_descrip = f'L:{np.round(loss.item(), 2)}'

        iterator.set_description(iter_descrip)
        loss.backward()
        optimizer.step()

    return losses

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
  else: #innate_immunity
    pass #TODO: add in the data loading here
  print(f'Done loading in  {args.data}\n')
  
  # TODO: add functionality for scvi and linearscvi
  
  # set seeds
  torch.manual_seed(args.seed)
  np.random.seed(args.seed)
  random.seed(args.seed) 
  
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
  
  
  ## Train the Model ##
  val_split = 0
  
  config = {
    "learning_rate": 0.005,
    "epochs": args.epochs, 
    "batch_size": 300,
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
  
  print('Initializing wandb and training model...')
  wandb.init(project="scvi-ablation", entity="ml-at-cl", name = model_name, config = config)

  if torch.cuda.is_available():
    Y = Y.cuda()
    gplvm = gplvm.cuda()
    gplvm.X_covars = gplvm.X_covars.cuda()
    likelihood = likelihood.cuda()
    gplvm.X_latent = gplvm.X_latent.cuda()
    
  losses = train(gplvm=gplvm, likelihood=likelihood, Y=Y, encoder = args.encoder,
               seed = config['seed'], learn_scale = config['learn_scale'], 
                lr=config['learning_rate'], epochs=config['epochs'], batch_size=config['batch_size']) 

  torch.save(losses, f'{model_dir}/{model_name}_losses.pt')
  torch.save(likelihood.state_dict(), f'{model_dir}/{model_name}_likelihood_state_dict.pt')
  torch.save(gplvm.state_dict(), f'{model_dir}/{model_name}_gplvm_state_dict.pt')

  wandb.finish()
  print('Done.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate metrics of specified train model on specified data")
    parser.add_argument('-m', '--model', type=str, help='overarching model', 
    					default = 'gplvm',
              choices = ['scvi', 'linear_scvi', 'gplvm'])
    parser.add_argument('-d', '--data', type=str, help='Data your model was trained on', 
              default = 'covid_data',
    					choices = ['covid_data', 'innate_immunity', 'splatter_nb'])
    parser.add_argument('-p', '--preprocessing', type = str, help='preprocessing of raw counts',
    					default = 'rawcounts',
    					choices = ['rawcounts', 
                        'libnormlogtrans', 
                        'logtranscolumnstd', 
                        'libnormlogtranscolumnstd'])
    parser.add_argument('-e', '--encoder', type = str, help='type of encoder',
    					default = 'scaly',
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
    parser.add_argument('--epochs', type = int, help = 'number of epochs to run',
                        default = 15)
    
    args = parser.parse_args()

    main(args)
    
