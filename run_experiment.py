import torch
import numpy as np
import scanpy as sc

import argparse
import wandb

import matplotlib.pyplot as plt

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
######################################
# Argparser 

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--model_name', help = "Model name")
parser.add_argument('-ld','--likelihood_distribution', default='NB', help='Likelihood distribution')
parser.add_argument('-df','--dispersion_factor', default=-1, help='Dispersion factor for NB') # default, learn the thetas
parser.add_argument('-iscale', '--learn_scale', default=True, help="Include scale factor in mean")

parser.add_argument('-ibc','--include_batch_norm', default=True, help='True or False for including batch norm')

args = parser.parse_args()

possible_likelihoods = ['NB', 'Poisson']

if(isinstance(args.dispersion_factor, int)):
    raise Exception('Please specify a positive integer for the dispersion factor or -1 if learnable theta or N/A.')
if(not (args.likelihood_distribution in possible_likelihoods)):
    raise Exception('Please specify a likelihood distribution that is either NB or Poisson.')

######################################
# Load in prepped data

import os, scvi
from gpytorch.priors import NormalPrior

data_dir = "data/COVID_Stephenson/"
adata = sc.read_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")

######################################
# Load in trained scVI and linear scVI models + their embeddings
######################################
# inialialize scvi encoder + linear gplvm model
import gpytorch
from model import GPLVM, LatentVariable, VariationalELBO, trange, BatchIdx, _KL
from utils.preprocessing import setup_from_anndata
from scvi.distributions import NegativeBinomial, Poisson

seed = 42
torch.manual_seed(seed)
softplus = torch.nn.Softplus()
softmax = torch.nn.Softmax(-1)

class PoissonLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self):
        super().__init__()

    def forward(self, function_samples, **kwargs):
        # if(args.learn_scale):
        scale = kwargs['scale'][:, 0] # set to S_l
        # else:
        #     scale = 1
        fs = function_samples.softmax(dim=-1) 
        return Poisson(rate=scale * fs)

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        log_prob_lambda = lambda function_samples: self.forward(function_samples, **kwargs).log_prob(observations)
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob



class NBLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self, d):
        super().__init__()
        # if(args.dispersion_factor == -1):
        self.log_theta = torch.nn.Parameter(torch.ones(d)) # <- learned via SVI

    def forward(self, function_samples, **kwargs):
        # if(args.learn_scale):
        scale = kwargs['scale'][:, 0] # set to S_l
        # else:
            # scale = 1
        fs = function_samples.softmax(dim=-1) 

        # if(args.dispersion_factor > 0):
        theta = self.log_theta.exp()[:, None]
        # else:
            # theta=args.dispersion_factor

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
        
        self.prior_x = NormalPrior(
            torch.zeros(1, latent_dim),
            torch.ones(1, latent_dim))
        
        self.z_nnet = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001), # TODO: check what this is for
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
            torch.nn.Linear(128, latent_dim*2),
        )

        self.l_nnet = torch.nn.Sequential(
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
        
        # x_kl = _KL(q_l, self.prior_l, Y.shape[0], self.input_dim)
        # self.update_added_loss_term('x_kl', x_kl)
        
        return q_x.rsample(), q_l.rsample()

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
            X_sample = torch.cat((X_l, gplvm.X_covars[batch_index]), axis=1)
            gplvm_dist = gplvm(X_sample)
            loss = -elbo_func(gplvm_dist, Y_batch.T, scale=S_l).sum()
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

# load in data
Y, X_covars = setup_from_anndata(adata, 
                                 layer='counts',
                                 categorical_covariate_keys=['sample_id'], 
                                 continuous_covariate_keys=None,
                                 scale_gex=False)
Y[Y > 100] = 100 # TODO: capping this because numerical issues (think relu millions = millions, exponentiate leads to exploding numbers)

q = 10
seed = 42

(n, d), q = Y.shape, q
period_scale = np.pi
X_latent = ScalyEncoder(q, d + X_covars.shape[1]) 

gplvm = GPLVM(n, d, q, 
              covariate_dim=len(X_covars.T),
              # n_inducing=q + len(X_covars.T) + 30, #+1 ,  # TODO: larger inducing number shouldn't increase performance
              n_inducing=q + len(X_covars.T) + 1,
              period_scale=np.pi,
              X_latent=X_latent,
              X_covars=X_covars,
              pseudotime_dim=False)

gplvm.intercept = gpytorch.means.ConstantMean()
gplvm.random_effect_mean = gpytorch.means.ZeroMean()
gplvm.covar_module = gpytorch.kernels.LinearKernel(q + len(X_covars.T))

if(args.likelihood_distribution == 'NB'):
    likelihood = NBLikelihood(d)
if(args.likelihood_distribution == 'Poisson'):
    likelihood = PoissonLikelihood()
if(args.likelihood_distribution == 'Bernoulli'):
    likelihood = gpytorch.likelihoods.BernoulliLikelihood()

if torch.cuda.is_available():
    Y = Y.cuda()
    gplvm = gplvm.cuda()
    gplvm.X_covars = gplvm.X_covars.cuda()
    likelihood = likelihood.cuda()
    gplvm.X_latent = gplvm.X_latent.cuda()

# model_name = args.model_name
changes = 'bernoulli'
model_name = f'linear_gplvm_{changes}' 
wandb.init(project="scvi-ablation", entity="ml-at-cl", name = model_name)

wandb.config = {
  "learning_rate": 0.005,
  "epochs": 30,
  "batch_size": 30
}

losses = train(gplvm=gplvm, likelihood=likelihood, Y=Y, lr=wandb.config['learning_rate'], epochs=wandb.config['epochs'], batch_size=wandb.config['batch_size']) # TODO: check if you can make this run faster


# # if os.path.exists('latent_sd.pt'):
torch.save(losses, f'models/{model_name}_losses.pt')
torch.save(gplvm.X_latent.state_dict(), f'models/{model_name}_state_dict.pt')

# # model_name = 'linear_gplvm_inducing_vars_add30'
# X_latent.load_state_dict(torch.load(f'models/{model_name}_state_dict.pt'))
# X_latent.eval()
# X_latent.cuda()

#-- plotting --
# find X_latent means directly
z_params = gplvm.X_latent.z_nnet(torch.cat([Y.cuda(), X_covars.cuda()], axis=1))
# z_params = X_latent.z_nnet(torch.cat([Y.cuda(), X_covars.cuda()], axis=1))

X_latent_dims =  z_params[..., :gplvm.X_latent.latent_dim]
# X_latent_dims = z_params[..., :X_latent.latent_dim]

adata.obsm[f'X_{model_name}_latent'] = X_latent_dims.detach().cpu().numpy()

sc.pp.neighbors(adata, n_neighbors=50, use_rep=f'X_{model_name}_latent', key_added=f'X_{model_name}_k50')
sc.tl.umap(adata, neighbors_key=f'X_{model_name}_k50')
adata.obsm[f'X_umap_{model_name}'] = adata.obsm['X_umap'].copy()

plt.rcParams['figure.figsize'] = [10,10]
col_obs = ['harmonized_celltype', 'Site']
sc.pl.embedding(adata, f'X_umap_{model_name}', color = col_obs, legend_loc='on data', size=5)

# import metrics stuff
import scib

batch_metrics = calc_batch_metrics(adata, embed_key = f'X_{model_name}_latent', batch_key = 'Site', label_key = 'harmonized_celltype')
torch.save(batch_metrics, f'models/{model_name}_batch_metrics.pt')

bio_metrics = calc_bio_metrics(adata, embed_key = f'X_{model_name}_latent')
torch.save(bio_metrics, f'models/{model_name}_bio_metrics.pt')

wandb.log({'batch_metrics': batch_metrics, 'bio_metrics': bio_metrics})


