import torch
import numpy as np
import scanpy as sc
import random

import argparse
import wandb

import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
######################################
# Load in prepped data

import os, scvi
from gpytorch.priors import NormalPrior

data_dir = "data/COVID_Stephenson/"
adata = sc.read_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")

######################################
# inialialize scvi encoder + linear gplvm model
import gpytorch
from model import GPLVM, LatentVariable, VariationalELBO, trange, BatchIdx, _KL, GaussianLikelihood
from utils.preprocessing import setup_from_anndata
from scvi.distributions import NegativeBinomial, ZeroInflatedNegativeBinomial
from torch.distributions import LogNormal
# import metrics

softplus = torch.nn.Softplus()
softmax = torch.nn.Softmax(-1)

class NBLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self, d):
        super().__init__()
        # self.log_theta = torch.nn.Parameter(torch.ones(d)) # learned theta

    def forward(self, function_samples, **kwargs):
        scale = kwargs['scale'][:, 0] # set to S_l, learned scaling factor
        # scale = 1 # fixed scale = 1

        fs = function_samples.softmax(dim=-1) 

        # theta = self.log_theta.exp()[:, None] # learned theta
        theta = 1 # fixed theta = 1
        
        return NegativeBinomial(
            mu=scale * fs,
            theta = theta,
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        log_prob_lambda = lambda function_samples: self.forward(function_samples, **kwargs).log_prob(observations)
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob

class ScalyEncoder(LatentVariable):
    """ KL added for both q_x and q_l"""
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim

        self._added_loss_terms['x_kl'] = None
        self._added_loss_terms['l_kl'] = None
        
        self.prior_x = NormalPrior(                 # prior for latent variable
            torch.zeros(1, latent_dim),
            torch.ones(1, latent_dim))
        
        self.prior_l = LogNormal(loc=0, scale=1)    # prior for scaling factor, may need to change with empirical mean and variance
        
        self.z_nnet = torch.nn.Sequential(          # NN for latent variables
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001), 
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
            torch.nn.Linear(128, latent_dim*2),
        )

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
        z_params = self.z_nnet(torch.cat([Y, X_covars], axis=1))
        l_params = self.l_nnet(torch.cat([Y, X_covars], axis=1))

        q_x = torch.distributions.Normal(
            z_params[..., :self.latent_dim],
            softplus(z_params[..., self.latent_dim:]) + 1e-4
        )
        q_l = torch.distributions.LogNormal(
            l_params[..., :1].tanh()*10,
            l_params[..., 1:].sigmoid()*10 + 1e-4
        )
        ## Adding KL(q|p) loss term 
        x_kl = _KL(q_x, self.prior_x, Y.shape[0], self.input_dim)
        self.update_added_loss_term('x_kl', x_kl)
        
        l_kl = _KL(q_l, self.prior_l, Y.shape[0], self.input_dim)
        self.update_added_loss_term('l_kl', l_kl)
        
        return q_x.rsample() , q_l.rsample()

def train(gplvm, likelihood, Y, epochs=100, batch_size=100, lr=0.01):
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

        try:
            # ---------------------------------
            Y_batch = Y[batch_index]
            X_l, S_l = gplvm.X_latent(Y_batch, gplvm.X_covars[batch_index])
            # X_l = gplvm.X_latent(Y_batch, gplvm.X_covars[batch_index])     # use this when scaling factor is not learned
            X_sample = torch.cat((X_l, gplvm.X_covars[batch_index]), axis=1)
            gplvm_dist = gplvm(X_sample)
            loss = -elbo_func(gplvm_dist, Y_batch.T, scale=S_l).sum()
            # loss = -elbo_func(gplvm_dist, Y_batch.T).sum()                 # use this when scaling factor is not learned
            # ---------------------------------
        except:
            from IPython.core.debugger import set_trace; set_trace()
        losses.append(loss.item())
        wandb.log({'loss': loss.item()})
        iterator.set_description(f'L:{np.round(loss.item(), 2)}')
        loss.backward()
        optimizer.step()

    return losses
#############
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# load in data
Y, X_covars = setup_from_anndata(adata, 
                                 layer='counts',
                                 categorical_covariate_keys=['sample_id'], 
                                 continuous_covariate_keys=None,
                                 scale_gex=False)

q = 10
(n, d)= Y.shape
period_scale = np.pi
X_latent = ScalyEncoder(q, d + X_covars.shape[1])

# X_latent =  Linear1LayerEncoder(q, d + X_covars.shape[1])
# X_latent =  Linear2LayerEncoder(q, d + X_covars.shape[1])
# X_latent =  OneLayerEncoder(q, d + X_covars.shape[1])


gplvm = GPLVM(n, d, q, 
              covariate_dim=len(X_covars.T),
              # n_inducing=q + len(X_covars.T) + 1 + 30, #+1 ,  # TODO: larger inducing number shouldn't increase performance
              n_inducing=q + len(X_covars.T) + 1,
              period_scale=np.pi,
              X_latent=X_latent,
              X_covars=X_covars,
              pseudotime_dim=False)

gplvm.intercept = gpytorch.means.ConstantMean()
gplvm.random_effect_mean = gpytorch.means.ZeroMean()
gplvm.covar_module = gpytorch.kernels.LinearKernel(q + len(X_covars.T))

likelihood = NBLikelihood(d)
# likelihood = gpytorch.likelihoods.BernoulliLikelihood()
# likelihood = GaussianLikelihood(batch_shape=gplvm.batch_shape)

if torch.cuda.is_available():
    Y = Y.cuda()
    gplvm = gplvm.cuda()
    gplvm.X_covars = gplvm.X_covars.cuda()
    likelihood = likelihood.cuda()
    gplvm.X_latent = gplvm.X_latent.cuda()

# model_name = args.model_name
changes = 'nblikelihood_learnedtheta_learnedscale'
model_name = f'linear_gplvm_{changes}' 

config = {
  "learning_rate": 0.005,
  "epochs": 15, 
  "batch_size": 300,
  "likelihood": f'NBLikelihood({d}) with theta = 1 and learned scale', #f'GaussianLikelihood(batch_shape={gplvm.batch_shape})'
  'X_latent': gplvm.X_latent,
  'n_inducing': q + len(X_covars.T) + 1,
  'covariate_dim': len(X_covars.T),
  'period_scale': np.pi,
  'X_covars': X_covars,
  'pseudotime_dim': False,
  'seed': 42,
  'elbo_func': f'VariationalELBO(likelihood, gplvm, num_data={n}), learning inducing loc',
  'gplvm': gplvm,
}

wandb.init(project="scvi-ablation", entity="ml-at-cl", name = model_name, config = config)

losses = train(gplvm=gplvm, likelihood=likelihood, Y=Y, lr=config['learning_rate'], epochs=config['epochs'], batch_size=config['batch_size']) 


# bio_metrics = torch.load('models/linear_gplvm_nblikelihood_learned_dispersion_bio_metrics.pt')

torch.save(losses, f'models/{model_name}_losses.pt')
torch.save(gplvm.X_latent.state_dict(), f'models/{model_name}_state_dict.pt')

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
X_latent.load_state_dict(torch.load(f'models/{model_name}_state_dict.pt'))
X_latent.eval()

# # find X_latent means directly
# z_params = gplvm.X_latent.z_nnet(torch.cat([Y, X_covars], axis=1))
z_params = X_latent.z_nnet(torch.cat([Y, X_covars], axis=1))

# X_latent_dims =  z_params[..., :gplvm.X_latent.latent_dim]
X_latent_dims = z_params[..., :X_latent.latent_dim]

adata.obsm[f'X_{model_name}_latent'] = X_latent_dims.detach().numpy()

#-- plotting --
# sc.pp.neighbors(adata, n_neighbors=50, use_rep=f'X_{model_name}_latent', key_added=f'X_{model_name}_k50')
# sc.tl.umap(adata, neighbors_key=f'X_{model_name}_k50')
# adata.obsm[f'X_umap_{model_name}'] = adata.obsm['X_umap'].copy()

# plt.rcParams['figure.figsize'] = [10,10]
# col_obs = ['harmonized_celltype', 'Site']
# sc.pl.embedding(adata, f'X_umap_{model_name}', color = col_obs, legend_loc='on data', size=5)
#------
# def test(adata, batch_key, label_key, embed_key):
#     import ipdb; ipdb.set_trace()
#     temp_score = scib.me.ilisi_graph(adata, batch_key=batch_key, type_="embed", use_rep=embed_key)
#     print(temp_score)
#     return temp_score
# test(adata, batch_key = 'Site', label_key = 'harmonized_celltype', embed_key =  f'X_{model_name}_latent')



# # import metrics stuff
import scib
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
