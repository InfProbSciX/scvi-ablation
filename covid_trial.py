
import torch
import numpy as np
import scanpy as sc

import matplotlib.pyplot as plt
plt.ion(); plt.style.use('seaborn-pastel')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#####################################################################
# Data Prep

import scvi

data_dir = "data/COVID_Stephenson/"

if False:
    from utils.initialise_latent_var import get_CC_effect_init, cc_genes

    adata_full = sc.read_h5ad(data_dir + "Stephenson.h5ad")
    adata_full.obs = adata_full.obs[['sample_id', 'Site', 'harmonized_celltype']].copy()

    adata_full.obs['cell_cycle_init'] = get_CC_effect_init(adata_full, cc_genes)
    adata = sc.pp.subsample(adata_full, n_obs = 100000, copy=True)

    adata.layers['counts'] = adata.layers['raw'].copy()
    sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)
    adata.write_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")

adata = sc.read_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")

#####################################################################
# Linear SCVI Run

adata_ref = adata.copy()
scvi.model.LinearSCVI.setup_anndata(adata_ref, batch_key='sample_id', layer='counts')

arches_params = dict(
    use_layer_norm="both",
    use_batch_norm="none",
    encode_covariates=True,
    dropout_rate=0.2,
    n_layers=2,
)

vae_ref = scvi.model.LinearSCVI(adata_ref, **arches_params)
vae_ref.train()

adata_ref.obsm["X_scVI"] = vae_ref.get_latent_representation()
adata.obsm["X_scVI"] = vae_ref.get_latent_representation(adata_ref)
sc.pp.neighbors(adata, n_neighbors=50, use_rep='X_scVI', key_added='scVI')
sc.tl.umap(adata, neighbors_key='scVI')
adata.obsm['X_umap_scVI'] = adata.obsm['X_umap'].copy()

plt.rcParams['figure.figsize'] = [10,10]
col_obs = ['harmonized_celltype', 'Site']
sc.pl.embedding(adata, 'X_umap_scVI', color = col_obs, legend_loc='on data', size=5)

#####################################################################
# Linear GPLVM Run

import gpytorch
from model import GPLVM, LatentVariable, VariationalELBO, trange, BatchIdx
from train import train_GPLVM, outputs_to_anndata
from utils.preprocessing import setup_from_anndata
from scvi.distributions import NegativeBinomial

Y, X_covars = setup_from_anndata(adata, 
                                 layer='counts',
                                 categorical_covariate_keys=['sample_id'], 
                                 continuous_covariate_keys=None,
                                 scale_gex=False)
Y[Y > 100] = 100

q = 10
batch_size = 500 ## how much does the RAM allow here?
seed = 42

torch.manual_seed(seed)
softplus = torch.nn.Softplus()
softmax = torch.nn.Softmax(-1)

class NBLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self, d):
        super().__init__()
        self.log_theta = torch.nn.Parameter(torch.ones(d))

    def forward(self, function_samples, **kwargs):
        scale = kwargs['scale'][:, 0]
        fs = function_samples.softmax(dim=-1)
        return NegativeBinomial(
            mu=scale * fs,
            theta=self.log_theta.exp()[:, None],
            scale=fs
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
            torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
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
              n_inducing=q + len(X_covars.T) + 1, 
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

losses = train(gplvm=gplvm, likelihood=likelihood, Y=Y, lr=0.005, epochs=80, batch_size=250)

# torch.save(gplvm.X_latent.state_dict(), 'latent_sd.pth')
gplvm.X_latent.load_state_dict(torch.load('latent_sd.pth'))

num_mc = 15
X_latent_dims = gplvm.X_latent(Y, gplvm.X_covars)[0]
for _ in range(num_mc - 1):
    X_latent_dims += gplvm.X_latent(Y, gplvm.X_covars)[0]
X_latent_dims /= num_mc

adata.obsm['X_BGPLVM_latent'] = X_latent_dims.detach().cpu()

sc.pp.neighbors(adata, n_neighbors=50, use_rep='X_BGPLVM_latent', key_added='BGPLVM')
sc.tl.umap(adata, neighbors_key='BGPLVM')
adata.obsm['X_umap_BGPLVM'] = adata.obsm['X_umap'].copy()

plt.rcParams['figure.figsize'] = [10,10]
col_obs = ['harmonized_celltype', 'Site']
sc.pl.embedding(adata, 'X_umap_BGPLVM', color = col_obs, legend_loc='on data', size=5)
