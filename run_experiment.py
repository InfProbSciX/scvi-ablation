import torch
import numpy as np
import scanpy as sc
import random

import argparse
import wandb
import matplotlib.pyplot as plt
plt.ion(); plt.style.use('seaborn-pastel')


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
######################################
# Load in prepped data

import os, scvi

data_dir = "data/COVID_Stephenson/"
adata = sc.read_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")

######################################
# inialialize scvi encoder + linear gplvm model
import gpytorch
from tqdm import trange
from model import GPLVM, LatentVariable, VariationalELBO, BatchIdx, _KL, PointLatentVariable
from utils.preprocessing import setup_from_anndata
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import NormalPrior
from torch.distributions import LogNormal


softplus = torch.nn.Softplus()
softmax = torch.nn.Softmax(-1)

## define likelihoods ##
class NBLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self, d, learn_scale, learn_theta):
        super().__init__()
        self.log_theta = torch.nn.Parameter(torch.ones(d)) # learned theta
        self.learn_scale = learn_scale
        self.learn_theta = learn_theta

    def forward(self, function_samples, **kwargs):
        fs = function_samples.softmax(dim=-1) 

        if(self.learn_scale):
            scale = kwargs['scale'][:, 0] # set to S_l, learned scaling factor
        else:
            scale = 1 # fixed scale = 1

        if(self.learn_theta):
            theta = self.log_theta.exp()[:, None] # learned theta
        else:
            theta = 1
        
        return NegativeBinomial(
            mu=scale * fs,
            theta = theta,
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        log_prob_lambda = lambda function_samples: self.forward(function_samples, **kwargs).log_prob(observations)
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob

## define encoders ##
class ScalyEncoder(LatentVariable):
    """ KL added for both q_x and q_l"""
    def __init__(self, latent_dim, input_dim, learn_scale, Y): # n_layers, layer_type):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.learn_scale = learn_scale
        
        self.prior_x = NormalPrior(                 # prior for latent variable
            torch.zeros(1, latent_dim),
            torch.ones(1, latent_dim))
        
        if(self.learn_scale):
            log_empirical_total_mean = float(torch.mean(torch.log(Y.sum(1))))
            log_empirical_total_var = float(torch.std(torch.log(Y.sum(1))))
            self.prior_l = LogNormal(loc=log_empirical_total_mean, scale=log_empirical_total_var)  

        self.register_added_loss_term("x_kl")    # register added loss terms
        if(self.learn_scale): 
            self.register_added_loss_term("l_kl")

        self.z_nnet = torch.nn.Sequential(          # NN for latent variables
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001), 
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
            torch.nn.Linear(128, latent_dim*2),
        )

        if(self.learn_scale):
            self.l_nnet = torch.nn.Sequential(          # NN for scaling factor
                torch.nn.Linear(input_dim, 128),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
                torch.nn.Linear(128, 128),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
                torch.nn.Linear(128, 1*2),
            )

    def forward(self, Y=None, X_covars=None):
        # z_params = self.z_nnet(torch.cat([Y, X_covars], axis=1))
        z_params = self.z_nnet(Y)

        q_x = torch.distributions.Normal(
            z_params[..., :self.latent_dim],
            softplus(z_params[..., self.latent_dim:]) + 1e-4
        )

        ## Adding KL(q|p) loss term 
        x_kl = _KL(q_x, self.prior_x, Y.shape[0], self.input_dim)
        self.update_added_loss_term('x_kl', x_kl)
        

        if(self.learn_scale):
            l_params = self.l_nnet(torch.cat([Y, X_covars], axis=1))
            
            q_l = torch.distributions.LogNormal(
                l_params[..., :1].tanh()*10,
                l_params[..., 1:].sigmoid()*10 + 1e-4
            )   

            ## Adding KL(q|p) loss term 
            l_kl = _KL(q_l, self.prior_l, Y.shape[0], self.input_dim)
            self.update_added_loss_term('l_kl', l_kl)
        
        if(self.learn_scale):
            return q_x.rsample(), q_l.rsample()
        return q_x.rsample()


## define training and validation functions
def evaluate(gplvm, likelihood, Y, learn_scale, val_indices, batch_size):
    n_val = len(val_indices)

    val_elbo_func = VariationalELBO(likelihood, gplvm, num_data=n_val)

    val_iterator = trange(int(np.ceil(n_val/batch_size)), leave = False)
    val_idx = BatchIdx(n_val, batch_size).idx()

    val_loss = 0
    with torch.no_grad():
        gplvm.eval()
        gplvm.X_latent.eval()
        for i in val_iterator:
            batch_index = val_indices[next(val_idx)]
            try:
                # ---------------------------------
                Y_batch = Y[batch_index]
                if(learn_scale):
                    X_l, S_l = gplvm.X_latent(Y_batch, gplvm.X_covars[batch_index])
                else:
                    X_l = gplvm.X_latent(Y_batch, gplvm.X_covars[batch_index])   
                X_sample = torch.cat((X_l, gplvm.X_covars[batch_index]), axis=1)
                gplvm_dist = gplvm(X_sample)
                if(learn_scale):
                    val_loss += -val_elbo_func(gplvm_dist, Y_batch.T, scale=S_l).sum() * len(batch_index)
                else:
                    val_loss += -val_elbo_func(gplvm_dist, Y_batch.T).sum() * len(batch_index)     
                # ---------------------------------
            except:
                from IPython.core.debugger import set_trace; set_trace()
    return val_loss/n_val # dividing by n_val to keep it as roughly average loss per datapoint, rather than summing it all

def train(gplvm, likelihood, Y, learn_scale, 
            seed = 42, epochs=100, batch_size=100, lr=0.01, 
            val_split = 0.2, eval_iter = 50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # set up train and validation splits
    n = len(Y)
    indices = list(range(n))
    np.random.shuffle(indices)
    split = int(np.floor(val_split * n))
    train_indices, val_indices = indices[split:], indices[:split]
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)

    n_train = len(train_indices)
    train_steps = int(np.ceil(epochs*n_train/batch_size))
    
    elbo_func = VariationalELBO(likelihood, gplvm, num_data=n_train)
    optimizer = torch.optim.Adam([
        dict(params=gplvm.parameters(), lr=lr),
        dict(params=likelihood.parameters(), lr=lr)
    ])

    train_losses = []; val_losses = []
    train_idx = BatchIdx(n_train, batch_size).idx()
    train_iterator = trange(train_steps, leave=False)
    for i in train_iterator:
        batch_index = train_indices[next(train_idx)]
        optimizer.zero_grad()

        try:
            # ---------------------------------
            Y_batch = Y[batch_index]
            if(learn_scale):
                # X_l, S_l = gplvm.X_latent(Y_batch, gplvm.X_covars[batch_index])
                X_l, S_l = gplvm.X_latent(Y_batch)
            else:
                # X_l = gplvm.X_latent(Y_batch, gplvm.X_covars[batch_index])     # use this when scaling factor is not learned
                # X_l = gplvm.X_latent(Y_batch)
                X_l = gplvm.X_latent(batch_index = batch_index, Y = Y)
            X_sample = torch.cat((X_l, gplvm.X_covars[batch_index]), axis=1)
            gplvm_dist = gplvm(X_sample)
            if(learn_scale):
                train_loss = -elbo_func(gplvm_dist, Y_batch.T, scale=S_l).sum()
            else:
                train_loss = -elbo_func(gplvm_dist, Y_batch.T).sum()                 # use this when scaling factor is not learned
            # ---------------------------------
        except:
            from IPython.core.debugger import set_trace; set_trace()
        train_losses.append(train_loss.item())
        wandb.log({'train loss': train_loss.item()})
        iter_descrip = f'L:{np.round(train_loss.item(), 2)}'

        # check validation loss
        if((val_split > 0) and (i % eval_iter == 0)):
            val_loss = evaluate(gplvm, likelihood, Y, learn_scale, val_indices, batch_size)
            val_losses.append(val_loss.item())
            wandb.log({'validation loss': val_loss.item()})
            iter_descrip = iter_descrip + f'; V:{np.round(val_loss.item(), 2)}'
            gplvm.train()
            gplvm.X_latent.train()

        train_iterator.set_description(iter_descrip)
        train_loss.backward()
        optimizer.step()

    return train_losses, val_losses
#############
X_cat_covars_keys = ['sample_id']
X_cts_covars_keys = None
X_covars_keys = X_cat_covars_keys 
# load in data
Y, X_covars = setup_from_anndata(adata, 
                                 layer='counts',
                                 categorical_covariate_keys=X_cat_covars_keys,
                                 continuous_covariate_keys=None,
                                 scale_gex=False)
# # normalize Y and log(x + 1) the counts
# Y_normalized = Y / Y.sum(1, keepdim = True) * 10000
# Y_log_normalized = torch.log(Y_normalized + 1)
# Y_log = torch.log(Y + 1)
# Y = Y_normalized
# Y = Y_log
# Y_log_transformed = torch.log(Y + 1) #torch.log(Y/100 + 1)
# Y = Y_log_transformed/torch.std(Y_log_transformed, axis = 0) # don't need to consider mean because it's captured in Gaussian Likelihood


seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
q = 10
(n, d)= Y.shape
period_scale = np.Inf # no period/cell-cycle shenanigans

learn_scale = False
learn_theta = True

## Declare encoder ##
# X_latent = ScalyEncoder(q, d + X_covars.shape[1], learn_scale = learn_scale, Y= Y)
# X_latent = ScalyEncoder(q, d , learn_scale = learn_scale, Y= Y)
X_latent = PointLatentVariable(torch.randn(n, q))

# X_latent =  Linear1LayerEncoder(q, d + X_covars.shape[1])
# X_latent =  Linear2LayerEncoder(q, d + X_covars.shape[1])
# X_latent =  OneLayerEncoder(q, d + X_covars.shape[1])

n_inducing = q + len(X_covars.T) + 1

gplvm = GPLVM(n, d, q, 
              covariate_dim=len(X_covars.T),
              n_inducing=n_inducing,
              period_scale=period_scale,
              X_latent=X_latent,
              X_covars=X_covars,
              pseudotime_dim=False)

gplvm.intercept = gpytorch.means.ConstantMean()
gplvm.random_effect_mean = gpytorch.means.ZeroMean()
gplvm.covar_module = gpytorch.kernels.LinearKernel(q + len(X_covars.T))

## Declare likelihood ##
likelihood = NBLikelihood(d, learn_scale = learn_scale, learn_theta = learn_theta)
# likelihood = GaussianLikelihood(batch_shape=gplvm.batch_shape)

if torch.cuda.is_available():
    Y = Y.cuda()
    gplvm = gplvm.cuda()
    gplvm.X_covars = gplvm.X_covars.cuda()
    likelihood = likelihood.cuda()
    gplvm.X_latent = gplvm.X_latent.cuda()

# # model_name = args.model_name
# changes = 'gaussianlikelihood_lognormalizedandstdstandardized'
# model_name = 'gplvm_rawcounts_scaly_linear_sampleid_nocc_nblikelihood'
# model_name = 'gplvm_rawcounts_scalynocovars_linear_sampleid_nocc_nblikelihood'
model_name = 'gplvm_rawcounts_point_linear_sampleid_nocc_nblikelihood'
if(learn_scale):
    model_name = model_name + 'learnscale'
else:
    model_name = model_name + 'noscale'
if(learn_theta):
    model_name = model_name + 'learntheta'
else:
    model_name = model_name + 'fixedtheta1'


val_split = 0

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

config = {
  "learning_rate": 0.005,
  "epochs": 15, 
  "batch_size": 300,
  'likelihood':f'NBLikelihood(d, learn_scale = {learn_scale}, learn_theta = {learn_theta})',
  # "likelihood": f'GaussianLikelihood(batch_shape={gplvm.batch_shape})',#
  'X_latent': gplvm.X_latent,
  'n_inducing': q + len(X_covars.T) + 1,
  'covariate_dim': len(X_covars.T),
  'validation_split': val_split,
  'eval_iter': 100,
  'period_scale': period_scale,
  'X_covars': X_covars,
  'X_covar_keys': X_covars_keys,
  'pseudotime_dim': False,
  'seed': seed,
  'elbo_func': f'VariationalELBO(likelihood, gplvm, num_data={n - int(np.floor(n*val_split))}), learning inducing loc',
  'gplvm': gplvm,
  'learn_scale': learn_scale,
}

wandb.init(project="scvi-ablation", entity="ml-at-cl", name = model_name, config = config)

# losses = train(gplvm=gplvm, likelihood=likelihood, Y=Y,
#                epochs=config['epochs'], batch_size=config['batch_size'], lr=config['learning_rate']) 
losses = train(gplvm=gplvm, likelihood=likelihood, Y=Y,seed = config['seed'], 
                learn_scale = config['learn_scale'], 
                val_split = config['validation_split'], eval_iter = config['eval_iter'], 
                lr=config['learning_rate'], epochs=config['epochs'], batch_size=config['batch_size']) 


# bio_metrics = torch.load('models/linear_gplvm_nblikelihood_learned_dispersion_bio_metrics.pt')

torch.save(losses, f'models/{model_name}_losses.pt')
# torch.save(gplvm.X_latent.state_dict(), f'models/{model_name}_Xlatent_state_dict.pt')
torch.save(gplvm.X_latent.X.detach(), f'models/{model_name}_X.pt')
torch.save(gplvm.state_dict(), f'models/{model_name}_gplvm_state_dict.pt')


Y = Y.cpu()
X_covars = X_covars.cpu()
gplvm = gplvm.cpu()
gplvm.X_covars = gplvm.X_covars.cpu()
likelihood = likelihood.cpu()
gplvm.X_latent = gplvm.X_latent.cpu()
torch.cuda.empty_cache()
import gc
gc.collect()
######
loading_in = True
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# model_name = ''
model_name = 'original_amortized_gplvm_lognormalizedandstdstandardized'


if(loading_in):
    gplvm.X_latent.load_state_dict(torch.load(f'models/{model_name}_X_latent_state_dict.pt'))
    gplvm.X_latent.eval()
    gplvm.load_state_dict(torch.load(f'models/{model_name}_gplvm_state_dict.pt'))

    # z_params = X_latent.nnet(Y)
    z_params = X_latent.z_nnet(torch.cat([Y, X_covars], axis=1))
    X_latent_dims = z_params [..., :X_latent.latent_dim]
else:
    z_params = gplvm.X_latent.z_nnet(torch.cat([Y, X_covars], axis=1))
    X_latent_dims = z_params[..., :gplvm.X_latent.latent_dim]


adata.obsm[f'X_{model_name}_latent'] = X_latent_dims.detach().numpy()

## -- plotting --
sc.pp.neighbors(adata, n_neighbors=50, use_rep=f'X_{model_name}_latent', key_added=f'X_{model_name}_k50')
sc.tl.umap(adata, neighbors_key=f'X_{model_name}_k50')
adata.obsm[f'X_umap_{model_name}'] = adata.obsm['X_umap'].copy()

plt.rcParams['figure.figsize'] = [10,10]
col_obs = ['harmonized_celltype', 'Site']
sc.pl.embedding(adata, f'X_umap_{model_name}', color = col_obs, legend_loc='on data', size=5)
## ------
# def test(adata, batch_key, label_key, embed_key):
#     import ipdb; ipdb.set_trace()
#     temp_score = scib.me.ilisi_graph(adata, batch_key=batch_key, type_="embed", use_rep=embed_key)
#     print(temp_score)
#     return temp_score
# test(adata, batch_key = 'Site', label_key = 'harmonized_celltype', embed_key =  f'X_{model_name}_latent')

print(calc_rmse(gplvm, Y, n_trials = 1000))

# # import metrics stuff
from utils.metrics import calc_batch_metrics, calc_bio_metrics
batch_metrics = calc_batch_metrics(adata, embed_key = f'X_{model_name}_latent')#, metrics_list = [''])
torch.save(batch_metrics, f'models/{model_name}_batch_metrics.pt')

bio_metrics = calc_bio_metrics(adata, embed_key = f"X_{model_name}_latent")
torch.save(bio_metrics, f'models/{model_name}_bio_metrics.pt')

wandb.log({'batch_metrics': batch_metrics, 'bio_metrics': bio_metrics})
wandb.finish()

###################################
# # EXTRA STUFF
# class Linear1LayerEncoder(LatentVariable):
#     def __init__(self, latent_dim, input_dim):
#         super().__init__()
#         self.latent_dim = latent_dim

#         self.z_nnet = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 128),
#             torch.nn.Linear(128, latent_dim*2),
#         )

#     def forward(self, Y=None, X_covars=None):
#         z_params = self.z_nnet(torch.cat([Y, X_covars], axis=1))

#         q_x = torch.distributions.Normal(
#             z_params[..., :self.latent_dim],
#             softplus(z_params[..., self.latent_dim:]) + 1e-4
#         )
#         return q_x.rsample()

# class Linear2LayerEncoder(LatentVariable):
#     def __init__(self, latent_dim, input_dim):
#         super().__init__()
#         self.latent_dim = latent_dim

#         self.z_nnet = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 128),
#             # torch.nn.ReLU(),
#             # torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001), # TODO: check what this is for
#             torch.nn.Linear(128, 128),
#             # torch.nn.ReLU(),
#             # torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
#             torch.nn.Linear(128, latent_dim*2),
#         )

#     def forward(self, Y=None, X_covars=None):
#         z_params = self.z_nnet(torch.cat([Y, X_covars], axis=1))

#         q_x = torch.distributions.Normal(
#             z_params[..., :self.latent_dim],
#             softplus(z_params[..., self.latent_dim:]) + 1e-4
#         )
#         return q_x.rsample()

# class OneLayerEncoder(LatentVariable):
#     def __init__(self, latent_dim, input_dim):
#         super().__init__()
#         self.latent_dim = latent_dim

#         self.z_nnet = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 128),
#             torch.nn.ReLU(),
#             torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001), # TODO: check what this is for
#             # torch.nn.Linear(128, 128),
#             # torch.nn.ReLU(),
#             # torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
#             torch.nn.Linear(128, latent_dim*2),
#         )

#     def forward(self, Y=None, X_covars=None):
#         z_params = self.z_nnet(torch.cat([Y, X_covars], axis=1))

#         q_x = torch.distributions.Normal(
#             z_params[..., :self.latent_dim],
#             softplus(z_params[..., self.latent_dim:]) + 1e-4
#         )
#         return q_x.rsample()
