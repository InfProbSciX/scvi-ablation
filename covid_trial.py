
import torch
import numpy as np
import scanpy as sc

import matplotlib.pyplot as plt
import wandb
# plt.ion(); plt.style.use('seaborn-pastel')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#####################################################################
# Data Prep

import os, scvi

data_dir = "data/COVID_Stephenson/"

# if not os.path.exists(os.path.join(data_dir, "Stephenson.subsample.100k.h5ad")):
    # os.makedirs(data_dir, exist_ok=True)

    # import gdown
    # gdown.download(id='1Sw5UnLPRLD-4fyFItO4bQbmJioUjABvf', output=data_dir)

    # from utils.initialise_latent_var import get_CC_effect_init, cc_genes

    # adata_full = sc.read_h5ad(data_dir + "Stephenson.h5ad")
    # adata_full.obs = adata_full.obs[['sample_id', 'Site', 'harmonized_celltype']].copy()

    # adata_full.obs['cell_cycle_init'] = get_CC_effect_init(adata_full, cc_genes)
#     adata = sc.pp.subsample(adata_full, n_obs=100000, copy=True)

#     adata.layers['counts'] = adata.layers['raw'].copy()
#     # sample most variable genes
#     sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)
#     adata.write_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")

#     # TODO: what happens when you don't look at most variable genes? is it worth it?

adata = sc.read_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")
#####################################################################
# SCVI Run

adata_ref_scvi = adata.copy()
# scvi.model.SCVI.setup_anndata(adata_ref_scvi, batch_key='sample_id', layer='counts')

# arches_params = dict(
#     use_layer_norm="both",
#     use_batch_norm="none",
#     encode_covariates=True,
#     dropout_rate=0.2,
#     n_layers=2,
#     gene_likelihood = 'nb',
# )

# scvi_ref = scvi.model.SCVI(adata_ref_scvi, **arches_params)
# scvi_ref.train()

# adata_ref_scvi.obsm["X_nonlinear_nb_scVI"] = scvi_ref.get_latent_representation()
# adata.obsm["X_nonlinear_nb_scVI"] = scvi_ref.get_latent_representation(adata_ref_scvi)
# sc.pp.neighbors(adata, n_neighbors=50, use_rep='X_nonlinear_nb_scVI', key_added='nonlinear_nb_scVI')
# sc.tl.umap(adata, neighbors_key='nonlinear_scVI')
# adata.obsm['X_umap_nonlinear_scVI'] = adata.obsm['X_umap'].copy()

# plt.rcParams['figure.figsize'] = [10,10]
# col_obs = ['harmonized_celltype', 'Site']
# sc.pl.embedding(adata, 'X_umap_nonlinear_scVI', color = col_obs, legend_loc='on data', size=5)

# scvi_ref.save('models/nonlinearNBSCVI/')
scvi_ref = scvi.model.SCVI.load('models/nonlinearNBSCVI', adata_ref_scvi)

adata_ref_scvi.obsm["X_scVI"] = scvi_ref.get_latent_representation()
adata.obsm["X_scVI"] = scvi_ref.get_latent_representation(adata_ref_scvi)

# import metrics
# scvi_reconstruction_error = scvi_ref.get_reconstruction_error(adata_ref_scvi)
# print(scvi_reconstruction_error)

bio_metrics, _ = calc_bio_metrics(adata, embed_key = 'X_scVI')
torch.save(bio_metrics, 'models/nonlinearNBSCVI_bio_metrics.pt')
# wandb.log({'scVI_bio_metrics': bio_metrics})
batch_metrics = calc_batch_metrics(adata, embed_key = 'X_scVI')
torch.save(batch_metrics, 'models/nonlinearNBSCVI_batch_metrics.pt')

# umap stuff
sc.pp.neighbors(adata, n_neighbors=50, use_rep='X_scVI', key_added='scVI_k50')
sc.tl.umap(adata, neighbors_key='scVI_k50')
adata.obsm['X_umap_scVI'] = adata.obsm['X_umap'].copy()


plt.rcParams['figure.figsize'] = [10,10]
col_obs = ['harmonized_celltype', 'Site']
sc.pl.embedding(adata, 'X_umap_scVI', color = col_obs, legend_loc='on data', size=5)

#####################################################################
# Linear SCVI Run

adata_ref = adata.copy()
# scvi.model.LinearSCVI.setup_anndata(adata_ref, batch_key='sample_id', layer='counts')

# arches_params = dict(
#     use_layer_norm="both",
#     use_batch_norm="none",
#     encode_covariates=True,
#     dropout_rate=0.2,
#     n_layers=2,
# )

# vae_ref = scvi.model.LinearSCVI(adata_ref, **arches_params)
# vae_ref.train()

# # vae_ref.save('models/linearSCVI.pt')
linear_scvi_ref = scvi.model.LinearSCVI.load('models/linearSCVI.pt', adata_ref)

#-- plotting--

adata_ref.obsm["X_linear_scVI"] = linear_scvi_ref.get_latent_representation()
adata.obsm["X_linear_scVI"] = linear_scvi_ref.get_latent_representation(adata_ref)

bio_metrics, _ = calc_bio_metrics(adata, embed_key = 'X_linear_scVI')
torch.save(bio_metrics, 'models/linearNBSCVI_bio_metrics.pt')
# wandb.log({'scVI_bio_metrics': bio_metrics})
batch_metrics = calc_batch_metrics(adata, embed_key = 'X_linear_scVI')
torch.save(batch_metrics, 'models/linearNBSCVI_batch_metrics.pt')



sc.pp.neighbors(adata, n_neighbors=50, use_rep='X_linear_scVI', key_added='linear_scVI')
sc.tl.umap(adata, neighbors_key='linear_scVI')
adata.obsm['X_umap_linear_scVI'] = adata.obsm['X_umap'].copy()

plt.rcParams['figure.figsize'] = [10,10]
col_obs = ['harmonized_celltype', 'Site']
sc.pl.embedding(adata, 'X_umap_linear_scVI', color = col_obs, legend_loc='on data', size=5)
#####################################################################
# Original GPLVM Run
import gpytorch
from model import GPLVM, LatentVariable, VariationalELBO, trange, BatchIdx, PointLatentVariable, GaussianLikelihood, NNEncoder
from utils.preprocessing import setup_from_anndata
from scvi.distributions import NegativeBinomial

Y, X_covars = setup_from_anndata(adata, 
                                 layer='counts',
                                 categorical_covariate_keys=['sample_id'], 
                                 continuous_covariate_keys=None,
                                 scale_gex=False)
Y[Y > 100] = 100 # TODO: capping this because numerical issues (think relu millions = millions, exponentiate leads to exploding numbers)

q = 10
seed = 42

torch.manual_seed(seed)

X_latent = NNEncoder(n, q, d, (128, 128))

(n, d) = Y.shape
period_scale = np.pi

gplvm = GPLVM(n, d, q,
              covariate_dim=len(X_covars.T),
              n_inducing=len(X_covars.T)+1,
              period_scale=np.pi,
              X_latent=X_latent,
              X_covars=X_covars,
              pseudotime_dim=False
             )
likelihood = GaussianLikelihood(batch_shape=gplvm.batch_shape)

if torch.cuda.is_available():
    Y = Y.cuda()
    gplvm = gplvm.cuda()
    gplvm.X_covars = gplvm.X_covars.cuda()
    likelihood = likelihood.cuda()


def train(gplvm, likelihood, Y, epochs=100, batch_size=100, lr=0.005):

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

        # ---------------------------------
        Y_batch = Y[batch_index]
        X_sample = torch.cat((
                gplvm.X_latent(batch_index, Y_batch),
                gplvm.X_covars[batch_index]
            ), axis=1)
        gplvm_dist = gplvm(X_sample)
        loss = -elbo_func(gplvm_dist, Y_batch.T).sum()
        # ---------------------------------

        losses.append(loss.item())
        wandb.log({'loss': loss.item()})
        iterator.set_description(f'L:{np.round(loss.item(), 2)}')
        loss.backward()
        optimizer.step()

    return losses

gplvm.X_latent.X.requires_grad_(True)
model_name = f'original_amortized_gplvm' 
wandb.init(project="scvi-ablation", entity="ml-at-cl", name = model_name)

wandb.config = {
  "learning_rate": 0.005,
  "epochs": 30,
  "batch_size": 100
}
losses = train(gplvm=gplvm, likelihood=likelihood, Y=Y,
               epochs=wandb.config['epochs'], batch_size=wandb.config['batch_size'], lr=0.005) # 27min


# if os.path.exists('latent_sd.pt'):
torch.save(losses, f'models/{model_name}_losses.pt')
torch.save(gplvm.X_latent.state_dict(), f'models/{model_name}_statedict.pt')
# X_latent = ScalyEncoder(q, d + X_covars.shape[1]) 

#-- plotting --
# find X_latent means directly
z_params = gplvm.X_latent.nnet(Y)
# z_params = X_latent.z_nnet(torch.cat([Y.cuda(), X_covars.cuda()], axis=1))

X_latent_dims =  z_params[..., :gplvm.X_latent.latent_dim].tanh()*5
# X_latent_dims = z_params[..., :X_latent.latent_dim]

adata.obsm[f'X_{model_name}_latent'] = X_latent_dims.detach().cpu().numpy()


bio_metrics, __ = calc_bio_metrics(adata, embed_key = f'X_{model_name}_latent')
torch.save(bio_metrics, f'models/{model_name}_bio_metrics.pt')

batch_metrics = calc_batch_metrics(adata, embed_key = f'X_{model_name}_latent')
torch.save(batch_metrics, f'models/{model_name}_batch_metrics.pt')

sc.pp.neighbors(adata, n_neighbors=50, use_rep=f'X_{model_name}_latent', key_added=model_name)
sc.tl.umap(adata, neighbors_key=f'{model_name}')
adata.obsm[f'X_umap_{model_name}'] = adata.obsm['X_umap'].copy()

plt.rcParams['figure.figsize'] = [10,10]
col_obs = ['harmonized_celltype', 'Site']
sc.pl.embedding(adata, f'X_umap_{model_name}', color = col_obs, legend_loc='on data', size=5)

#####################################################################
# Linear GPLVM Run

import gpytorch
from model import GPLVM, LatentVariable, VariationalELBO, trange, BatchIdx
from utils.preprocessing import setup_from_anndata
from scvi.distributions import NegativeBinomial

Y, X_covars = setup_from_anndata(adata, 
                                 layer='counts',
                                 categorical_covariate_keys=['sample_id'], 
                                 continuous_covariate_keys=None,
                                 scale_gex=False)
Y[Y > 100] = 100 # TODO: capping this because numerical issues (think relu millions = millions, exponentiate leads to exploding numbers)

q = 10
seed = 42

torch.manual_seed(seed)
softplus = torch.nn.Softplus()
softmax = torch.nn.Softmax(-1)

class NBLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self, d):
        super().__init__()
        self.log_theta = torch.nn.Parameter(torch.ones(d)) # <- learned via SVI

    def forward(self, function_samples, **kwargs):
        scale = kwargs['scale'][:, 0] # set to S_l
        fs = function_samples.softmax(dim=-1) 
        return NegativeBinomial(
            mu=scale * fs,
            theta=self.log_theta.exp()[:, None],
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        log_prob_lambda = lambda function_samples: self.forward(function_samples, **kwargs).log_prob(observations)
        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob

class ScalyEncoder(LatentVariable):
    """ KL is ignored for now. """
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.latent_dim = latent_dim

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
        iterator.set_description(f'L:{np.round(loss.item(), 2)}')
        loss.backward()
        optimizer.step()

    return losses

(n, d), q = Y.shape, q
period_scale = np.pi
X_latent = ScalyEncoder(q, d + X_covars.shape[1]) 

gplvm = GPLVM(n, d, q, 
              covariate_dim=len(X_covars.T),
              n_inducing=q + len(X_covars.T) + 1,  # TODO: larger inducing number shouldn't increase performance
              period_scale=np.pi,
              X_latent=X_latent,
              X_covars=X_covars,
              pseudotime_dim=False)

gplvm.intercept = gpytorch.means.ConstantMean()
gplvm.random_effect_mean = gpytorch.means.ZeroMean()
gplvm.covar_module = gpytorch.kernels.LinearKernel(q + len(X_covars.T))

likelihood = NBLikelihood(d)

if torch.cuda.is_available():
    Y = Y.cuda()
    gplvm = gplvm.cuda()
    gplvm.X_covars = gplvm.X_covars.cuda()
    likelihood = likelihood.cuda()
    gplvm.X_latent = gplvm.X_latent.cuda()

batch_size = 100 ## how much does the RAM allow here?
n_epochs = 80 

losses = train(gplvm=gplvm, likelihood=likelihood, Y=Y, lr=0.005, epochs=n_epochs, batch_size=batch_size) # TODO: check if you can make this run faster

# if os.path.exists('latent_sd.pt'):
# torch.save(losses, 'models/latent_sd_noscale_losses.pt')
# torch.save(gplvm.X_latent.state_dict(), 'models/latent_sd_scaleexpt_statedict.pt')
X_latent = ScalyEncoder(q, d + X_covars.shape[1]) 
X_latent.load_state_dict(torch.load('models/latent_sd_scaleexpt_statedict.pt'))
X_latent.eval()
X_latent.cuda()

#-- plotting --
# find X_latent means directly
# z_params = gplvm.X_latent.z_nnet(torch.cat([Y.cuda(), X_covars.cuda()], axis=1))
z_params = X_latent.z_nnet(torch.cat([Y.cuda(), X_covars.cuda()], axis=1))

# X_latent_dims =  z_params[..., :gplvm.X_latent.latent_dim]
X_latent_dims = z_params[..., :X_latent.latent_dim]

adata.obsm['X_BGPLVM_latent'] = X_latent_dims.detach().cpu().numpy()


gplvm_bio_metrics, __ = calc_bio_metrics(adata, embed_key = 'X_BGPLVM_latent')
torch.save(gplvm_batch_metrics, 'models/gplvm_bio_metrics.pt')

gplvm_batch_metrics = calc_batch_metrics(adata, embed_key = 'X_BGPLVM_latent')
torch.save(gplvm_batch_metrics, 'models/gplvm_batch_metrics.pt')

sc.pp.neighbors(adata, n_neighbors=50, use_rep='X_BGPLVM_latent', key_added='BGPLVM')
sc.tl.umap(adata, neighbors_key='BGPLVM')
adata.obsm['X_umap_BGPLVM'] = adata.obsm['X_umap'].copy()

plt.rcParams['figure.figsize'] = [10,10]
col_obs = ['harmonized_celltype', 'Site']
sc.pl.embedding(adata, 'X_umap_BGPLVM', color = col_obs, legend_loc='on data', size=5)

