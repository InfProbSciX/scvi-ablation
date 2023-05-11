
import torch
import numpy as np
import random
import scanpy as sc

import matplotlib.pyplot as plt
import wandb
plt.ion(); plt.style.use('seaborn-pastel')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# #####################################################################
# # Data Prep for covid

# import os, scvi

# data_dir = "data/COVID_Stephenson/"

# # if not os.path.exists(os.path.join(data_dir, "Stephenson.subsample.100k.h5ad")):
# #     os.makedirs(data_dir, exist_ok=True)

# #     import gdown
# #     gdown.download(id='1Sw5UnLPRLD-4fyFItO4bQbmJioUjABvf', output=data_dir)

# #     from utils.initialise_latent_var import get_CC_effect_init, cc_genes

# #     adata_full = sc.read_h5ad(data_dir + "Stephenson.h5ad")
# #     adata_full.obs = adata_full.obs[['sample_id', 'Site', 'harmonized_celltype']].copy()

# #     adata_full.obs['cell_cycle_init'] = get_CC_effect_init(adata_full, cc_genes)
# #     adata = sc.pp.subsample(adata_full, n_obs=100000, copy=True)

# #     adata.layers['counts'] = adata.layers['raw'].copy()
# #     # sample most variable genes
# #     sc.pp.highly_variable_genes(adata, n_top_genes=5000, subset=True)
# #     adata.write_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")

# adata = sc.read_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")
#####################################################################
import os, scvi
data_dir = 'data/COVID_Stephenson/'
adata = sc.read_h5ad(data_dir + "Stephenson.subsample.100k.h5ad")

# import scipy.sparse as sp
# import pandas as pd
# data_dir = 'data/hipsci_ipscs/'

# X_covars = np.array(pd.read_csv(data_dir+'model_mat.csv')) # a lot more than just batch information I think
# X_covars = torch.tensor(X_covars).float()
# X_covars[:, 1] -= X_covars[:, 1].mean()
# X_covars[:, 1] /= X_covars[:, 1].std()

# Y = torch.tensor(sp.load_npz(f'{data_dir}/data.npz').T.todense())
# Y /= Y.std(axis=0)

#####################################################################
# SCVI Run
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

adata_ref_scvi = adata.copy()
scvi.model.SCVI.setup_anndata(adata_ref_scvi, batch_key='sample_id', layer='counts')

arches_params = dict(
    use_layer_norm="both",
    use_batch_norm="none",
    encode_covariates=True,
    dropout_rate=0.2,
    n_layers=2,
    gene_likelihood = 'nb',
)

# arches_params_zinb = dict(
#     use_layer_norm="both",
#     use_batch_norm="none",
#     encode_covariates=True,
#     dropout_rate=0.2,
#     n_layers=2,
#     gene_likelihood = 'zinb',
# )

train_params = dict(
    max_epochs=500, 
    train_size = 1.0,
    batch_size = 300,
    plan_kwargs = {'lr': 0.005}, 
)

scvi_ref = scvi.model.SCVI(adata_ref_scvi, **arches_params)
scvi_ref.train(**train_params)

# scvi_ref.save('models/nonlinearNBSCVI/')
scvi_ref.save('models/nonlinearNBSCVI/')
losses = scvi_ref.history['elbo_train']
torch.save(losses, f'models/nonlinearNBSCVI_losses.pt')

# scvi_ref = scvi.model.SCVI.load('models/nonlinearNBSCVI', adata_ref_scvi)

adata_ref_scvi.obsm["X_scVI"] = scvi_ref.get_latent_representation()
adata.obsm["X_scVI"] = scvi_ref.get_latent_representation(adata_ref_scvi)

from utils.metrics import calc_bio_metrics, calc_batch_metrics
bio_metrics = calc_bio_metrics(adata, embed_key = 'X_scVI')
torch.save(bio_metrics, 'models/nonlinearNBSCVI_bio_metrics.pt')

batch_metrics = calc_batch_metrics(adata, embed_key = 'X_scVI')
torch.save(batch_metrics, 'models/nonlinearNBSCVI_batch_metrics.pt')


## umap ##
umap_seed = 1
torch.manual_seed(umap_seed)
np.random.seed(umap_seed)
random.seed(umap_seed)

sc.pp.neighbors(adata, n_neighbors=50, use_rep='X_scVI', key_added='scVI_k50')
sc.tl.umap(adata, neighbors_key='scVI_k50', random_state = umap_seed)
adata.obsm['X_umap_scVI'] = adata.obsm['X_umap'].copy()


plt.rcParams['figure.figsize'] = [10,10]
col_obs = ['harmonized_celltype', 'Site']
sc.pl.embedding(adata, 'X_umap_scVI', color = col_obs, legend_loc='on data', size=5, save = 'umap_images/nonlinearNBSCVI_umap.png')

#####################################################################
# Linear SCVI Run
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

adata_ref = adata.copy()
X_covars_keys = 'sample_id'
scvi.data.setup_anndata(adata_ref, batch=X_covars_keys, layer='counts')

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

linearscvi = scvi.model.LinearSCVI(adata_ref, **arches_params)
linearscvi.train(**train_params)

linearscvi.save('models/linearNBSCVI/')
losses = linearscvi.history['elbo_train']
torch.save(losses, 'models/linearNBSCVI_losses.pt')
losses.plot()
plt.savefig("linearNBSCVI_losses.png")
plt.show()

ldvae = scvi.model.LinearSCVI.load('models/linearNBSCVI', adata_ref)

#-- plotting--

adata_ref.obsm["X_linear_scVI"] = linearscvi.get_latent_representation()
adata.obsm["X_linear_scVI"] = linearscvi.get_latent_representation(adata_ref)

bio_metrics = calc_bio_metrics(adata, embed_key = 'X_linear_scVI')
torch.save(bio_metrics, 'models/linearNBSCVI_bio_metrics.pt')
batch_metrics = calc_batch_metrics(adata, embed_key = 'X_linear_scVI')
torch.save(batch_metrics, 'models/linearNBSCVI_batch_metrics.pt')

## umap ##
umap_seed = 1
torch.manual_seed(umap_seed)
np.random.seed(umap_seed)
random.seed(umap_seed)

sc.pp.neighbors(adata, n_neighbors=50, use_rep='X_linear_scVI', key_added='linear_scVI')
sc.tl.umap(adata, neighbors_key='linear_scVI', random_state = umap_seed)
adata.obsm['X_umap_linear_scVI'] = adata.obsm['X_umap'].copy()

plt.rcParams['figure.figsize'] = [10,10]
col_obs = ['harmonized_celltype', 'Site']
sc.pl.embedding(adata, 'X_umap_linear_scVI', color = col_obs, legend_loc='on data', size=5, save = 'umap_images/linearNBSCVI_umap.png')
#####################################################################
# NNEncoder (hopefully original) GPLVM Run

import gpytorch
from model import GPLVM, LatentVariable,_KL,NormalPrior, softplus,VariationalELBO, trange, BatchIdx, PointLatentVariable, GaussianLikelihood #, NNEncoder
from utils.preprocessing import setup_from_anndata
from scvi.distributions import NegativeBinomial

X_covars_keys = ['sample_id'] # 'Site' -> check if sample ids are different for different sites, it does, so not including it
Y, X_covars = setup_from_anndata(adata, 
                                 layer='counts',
                                 categorical_covariate_keys=X_covars_keys, 
                                 continuous_covariate_keys=None,
                                 scale_gex=False)
# preprocessing
lib_norm = True
column_std = True
Y_temp = Y
if(lib_norm):
    Y_temp = Y_temp/Y_temp.sum(1, keepdim = True) * 10000
Y_temp = torch.log(Y_temp + 1)
if(column_std):
    Y_temp = Y_temp / torch.std(Y_temp, axis = 0) # don't need to consider mean because it's captured in Gaussian Likelihood
Y = Y_temp

q = 10
(n, d) = Y.shape
seed = 123

torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class NNEncoder(LatentVariable):    
    def __init__(self, n, latent_dim, data_dim, layers):
        super().__init__()
        self.n = n
        self.latent_dim = latent_dim
        self.prior_x = NormalPrior(
            torch.zeros(1, latent_dim),
            torch.ones(1, latent_dim))
        self.data_dim = data_dim
        self.latent_dim = latent_dim

        self._init_nnet(layers)
        self.register_added_loss_term("x_kl")

        self.jitter = torch.eye(latent_dim).unsqueeze(0)*1e-5

    def _init_nnet(self, hidden_layers):
        layers = (self.data_dim,) + hidden_layers + (self.latent_dim*2,)
        n_layers = len(layers)

        modules = []; last_layer = n_layers - 1
        for i in range(last_layer):
            modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
            if i < last_layer - 1: modules.append(softplus)

        self.nnet = torch.nn.Sequential(*modules)

    def forward(self, batch_index=None, Y=None):
        h = self.nnet(Y)
        mu = h[..., :self.latent_dim].tanh()*5
        sg = softplus(h[..., self.latent_dim:]) + 1e-4 # changed from + 1e-6

        q_x = torch.distributions.Normal(mu, sg)

        x_kl = _KL(q_x, self.prior_x, len(mu), self.data_dim)
        self.update_added_loss_term('x_kl', x_kl)
        return q_x.rsample()

class EncoderAdaptedFromSCVI(LatentVariable): # this is the same as the scalyencoder
    """ KL added for both q_x and q_l"""
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        
        self.prior_x = NormalPrior(                 # prior for latent variable
            torch.zeros(1, latent_dim),
            torch.ones(1, latent_dim))

        self.register_added_loss_term("x_kl")    # register added loss terms

        self.z_nnet = torch.nn.Sequential(          # NN for latent variables
            torch.nn.Linear(input_dim, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001), 
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
            torch.nn.Linear(128, latent_dim*2),
        )

    def forward(self, Y=None, X_covars = None):
        # z_params = self.z_nnet(Y)
        z_params = self.z_nnet(torch.cat([Y, X_covars], axis=1))

        q_x = torch.distributions.Normal(
            z_params[..., :self.latent_dim],
            softplus(z_params[..., self.latent_dim:]) + 1e-4
        )

        ## Adding KL(q|p) loss term 
        x_kl = _KL(q_x, self.prior_x, Y.shape[0], self.input_dim)
        self.update_added_loss_term('x_kl', x_kl)

        return q_x.rsample()

X_latent = NNEncoder(n, q, d, (128, 128))
# X_latent = EncoderAdaptedFromSCVI(q, d+ X_covars.shape[1])
# X_latent = PointLatentVariable(torch.randn(n, q))

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
gplvm.covar_module = gpytorch.kernels.LinearKernel(q + len(X_covars.T))

likelihood = GaussianLikelihood(batch_shape=gplvm.batch_shape)

# ## define training and validation functions
# def evaluate(gplvm, likelihood, Y, val_indices, batch_size):
#     n_val = len(val_indices)

#     val_elbo_func = VariationalELBO(likelihood, gplvm, num_data=n_val)

#     val_iterator = trange(int(np.ceil(n_val/batch_size)), leave = False)
#     val_idx = BatchIdx(n_val, batch_size).idx()

#     val_loss = 0
#     with torch.no_grad():
#         gplvm.eval()
#         gplvm.X_latent.eval()
#         for i in val_iterator:
#             batch_index = val_indices[next(val_idx)]
#             try:
#                 # ---------------------------------
#                 Y_batch = Y[batch_index]
#                 X_l = gplvm.X_latent()
#                 X_sample = torch.cat((X_l, gplvm.X_covars[batch_index]), axis=1)
#                 gplvm_dist = gplvm(X_sample)
#                 val_loss += -val_elbo_func(gplvm_dist, Y_batch.T).sum() * len(batch_index)     
#                 # ---------------------------------
#             except:
#                 from IPython.core.debugger import set_trace; set_trace()
#     return val_loss/n_val # dividing by n_val to keep it as roughly average loss per datapoint, rather than summing it all

def train(gplvm, likelihood, Y, 
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
            # X_l = gplvm.X_latent(Y = Y_batch) 
            # X_l = gplvm.X_latent(Y_batch, gplvm.X_covars[batch_index])     
            X_l = gplvm.X_latent(batch_index = batch_index, Y = Y)
            X_sample = torch.cat((X_l, gplvm.X_covars[batch_index]), axis=1)
            gplvm_dist = gplvm(X_sample)
            train_loss = -elbo_func(gplvm_dist, Y_batch.T).sum()                 
            # ---------------------------------
        except:
            from IPython.core.debugger import set_trace; set_trace()
        train_losses.append(train_loss.item())
        wandb.log({'train loss': train_loss.item()})
        iter_descrip = f'L:{np.round(train_loss.item(), 2)}'

        # check validation loss
        if((val_split > 0) and (i % eval_iter == 0)):
            val_loss = evaluate(gplvm, likelihood, Y, val_indices, batch_size)
            val_losses.append(val_loss.item())
            wandb.log({'validation loss': val_loss.item()})
            iter_descrip = iter_descrip + f'; V:{np.round(val_loss.item(), 2)}'
            gplvm.train()
            gplvm.X_latent.train()

        train_iterator.set_description(iter_descrip)
        train_loss.backward()
        optimizer.step()

    return train_losses, val_losses


model_name = 'gplvm_'
if(lib_norm):
    model_name = model_name + 'libnorm'
model_name = model_name + 'logtrans'
if(column_std):
    model_name = model_name + 'columnstd'
# model_name = model_name + f'_nnenc_linear_sampleid_nocc_gaussianlikelihood'
# model_name = model_name + f'_scalynocovars_linear_sampleid_nocc_gaussianlikelihood'
# model_name = model_name + f'_scaly_linear_sampleid_nocc_gaussianlikelihood'
model_name = model_name + f'_point_linear_sampleid_nocc_gaussianlikelihood'

val_split = 0
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

config = {
  "learning_rate": 0.005,
  "epochs": 10, 
  "batch_size": 300,
  "likelihood": f'GaussianLikelihood(batch_shape={gplvm.batch_shape})',
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
  'lib_norm': lib_norm,
  'column_std': column_std
}
wandb.finish()
wandb.init(project="scvi-ablation", entity="ml-at-cl", name = model_name, config = config)

if torch.cuda.is_available():
    Y = Y.cuda()
    gplvm = gplvm.cuda()
    gplvm.X_covars = gplvm.X_covars.cuda()
    likelihood = likelihood.cuda()

losses = train(gplvm=gplvm, likelihood=likelihood, Y=Y,seed = config['seed'], 
                val_split = config['validation_split'], eval_iter = config['eval_iter'], 
                lr=config['learning_rate'], epochs=config['epochs'], batch_size=config['batch_size'])


# if os.path.exists('latent_sd.pt'):
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
###
gplvm.X_latent.load_state_dict(torch.load(f'models/{model_name}_Xlatent_state_dict.pt', map_location=torch.device('cpu')))
gplvm.load_state_dict(torch.load(f'models/{model_name}_gplvm_state_dict.pt',  map_location=torch.device('cpu')))
gplvm.X_latent.eval()
gplvm.eval()

#-- plotting --
# find X_latent means directly
z_params = gplvm.X_latent.nnet(Y)
X_latent_dims = z_params[..., :gplvm.X_latent.latent_dim].tanh()*5
adata.obsm[f'X_{model_name}_latent'] = X_latent_dims.detach().cpu().numpy()
######
from utils.metrics import calc_batch_metrics, calc_bio_metrics
bio_metrics = calc_bio_metrics(adata, embed_key = f'X_{model_name}_latent')
torch.save(bio_metrics, f'models/{model_name}_bio_metrics.pt')

batch_metrics = calc_batch_metrics(adata, embed_key = f'X_{model_name}_latent')
torch.save(batch_metrics, f'models/{model_name}_batch_metrics.pt')

###########
# # calculate RMSE
# print(rmse)
# torch.save(rmse, f'models/{model_name}_rmse.pt')
###########

umap_seed = 1
torch.manual_seed(umap_seed)
np.random.seed(umap_seed)
random.seed(umap_seed)

sc.pp.neighbors(adata, n_neighbors=50, use_rep=f'X_{model_name}_latent', key_added=model_name)
sc.tl.umap(adata, neighbors_key=f'{model_name}', random_state = umap_seed)
adata.obsm[f'X_umap_{model_name}'] = adata.obsm['X_umap'].copy()

plt.rcParams['figure.figsize'] = [10,10]
col_obs = ['harmonized_celltype', 'Site']
sc.pl.embedding(adata, f'X_umap_{model_name}', color = col_obs, legend_loc='on data', 
    size=5, save=f'umap_images/{model_name}_umap.png')

# def train(gplvm, likelihood, Y, epochs=100, batch_size=100, lr=0.005):

#     n = len(Y)
#     steps = int(np.ceil(epochs*n/batch_size))
#     elbo_func = VariationalELBO(likelihood, gplvm, num_data=n)
#     optimizer = torch.optim.Adam([
#         dict(params=gplvm.parameters(), lr=lr),
#         dict(params=likelihood.parameters(), lr=lr)
#     ])

#     losses = []; idx = BatchIdx(n, batch_size).idx()
#     iterator = trange(steps, leave=False)
#     for i in iterator:
#         batch_index = next(idx)
#         optimizer.zero_grad()

#         # ---------------------------------
#         Y_batch = Y[batch_index]
#         X_sample = torch.cat((
#                 gplvm.X_latent(batch_index, Y_batch),
#                 gplvm.X_covars[batch_index]
#             ), axis=1)
#         gplvm_dist = gplvm(X_sample)
#         loss = -elbo_func(gplvm_dist, Y_batch.T).sum()
#         # ---------------------------------

#         losses.append(loss.item())
#         wandb.log({'loss': loss.item()})
#         iterator.set_description(f'L:{np.round(loss.item(), 2)}')
#         loss.backward()
#         optimizer.step()

#     return losses


# #####################################################################
# # Linear GPLVM Run

# import gpytorch
# from model import GPLVM, LatentVariable, VariationalELBO, trange, BatchIdx
# from utils.preprocessing import setup_from_anndata
# from scvi.distributions import NegativeBinomial

# Y, X_covars = setup_from_anndata(adata, 
#                                  layer='counts',
#                                  categorical_covariate_keys=['sample_id'], 
#                                  continuous_covariate_keys=None,
#                                  scale_gex=False)
# Y[Y > 100] = 100 # TODO: capping this because numerical issues (think relu millions = millions, exponentiate leads to exploding numbers)

# q = 10
# (n, d), q = Y.shape, q
# seed = 42

# torch.manual_seed(seed)
# softplus = torch.nn.Softplus()
# softmax = torch.nn.Softmax(-1)

# class NBLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
#     def __init__(self, d):
#         super().__init__()
#         self.log_theta = torch.nn.Parameter(torch.ones(d)) # <- learned via SVI

#     def forward(self, function_samples, **kwargs):
#         scale = kwargs['scale'][:, 0] # set to S_l
#         fs = function_samples.softmax(dim=-1) 
#         return NegativeBinomial(
#             mu=scale * fs,
#             theta=self.log_theta.exp()[:, None],
#         )

#     def expected_log_prob(self, observations, function_dist, *args, **kwargs):
#         log_prob_lambda = lambda function_samples: self.forward(function_samples, **kwargs).log_prob(observations)
#         log_prob = self.quadrature(log_prob_lambda, function_dist)
#         return log_prob

# class ScalyEncoder(LatentVariable):
#     """ KL is ignored for now. """
#     def __init__(self, latent_dim, input_dim):
#         super().__init__()
#         self.latent_dim = latent_dim

#         self.z_nnet = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 128),
#             torch.nn.ReLU(),
#             torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001), # TODO: check what this is for
#             torch.nn.Linear(128, 128),
#             torch.nn.ReLU(),
#             torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
#             torch.nn.Linear(128, latent_dim*2),
#         )

#         self.l_nnet = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 128),
#             torch.nn.ReLU(),
#             torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
#             torch.nn.Linear(128, 128),
#             torch.nn.ReLU(),
#             torch.nn.BatchNorm1d(128, momentum=0.01, eps=0.001),
#             torch.nn.Linear(128, 1*2),
#         )

#     def forward(self, Y=None, X_covars=None):
#         z_params = self.z_nnet(torch.cat([Y, X_covars], axis=1))
#         l_params = self.l_nnet(torch.cat([Y, X_covars], axis=1))

#         q_x = torch.distributions.Normal(
#             z_params[..., :self.latent_dim],
#             softplus(z_params[..., self.latent_dim:]) + 1e-4
#         )
#         q_l = torch.distributions.LogNormal(
#             l_params[..., :1].tanh()*10,
#             l_params[..., 1:].sigmoid()*10 + 1e-4
#         )
#         return q_x.rsample(), q_l.rsample()

# def train(gplvm, likelihood, Y, epochs=100, batch_size=100, lr=0.01):

#     n = len(Y)
#     steps = int(np.ceil(epochs*n/batch_size))
#     elbo_func = VariationalELBO(likelihood, gplvm, num_data=n)
#     optimizer = torch.optim.Adam([
#         dict(params=gplvm.parameters(), lr=lr),
#         dict(params=likelihood.parameters(), lr=lr)
#     ])

#     losses = []; idx = BatchIdx(n, batch_size).idx()
#     iterator = trange(steps, leave=False)
#     for i in iterator:
#         batch_index = next(idx)
#         optimizer.zero_grad()

#         try:
#             # ---------------------------------
#             Y_batch = Y[batch_index]
#             X_l, S_l = gplvm.X_latent(Y_batch, gplvm.X_covars[batch_index])
#             X_sample = torch.cat((X_l, gplvm.X_covars[batch_index]), axis=1)
#             gplvm_dist = gplvm(X_sample)
#             loss = -elbo_func(gplvm_dist, Y_batch.T, scale=S_l).sum()
#             # ---------------------------------
#         except:
#             from IPython.core.debugger import set_trace; set_trace()
#         losses.append(loss.item())
#         iterator.set_description(f'L:{np.round(loss.item(), 2)}')
#         loss.backward()
#         optimizer.step()

#     return losses

# (n, d), q = Y.shape, q
# period_scale = np.pi
# X_latent = ScalyEncoder(q, d + X_covars.shape[1]) 

# gplvm = GPLVM(n, d, q, 
#               covariate_dim=len(X_covars.T),
#               n_inducing=q + len(X_covars.T) + 1,  # TODO: larger inducing number shouldn't increase performance
#               period_scale=np.pi,
#               X_latent=X_latent,
#               X_covars=X_covars,
#               pseudotime_dim=False)

# gplvm.intercept = gpytorch.means.ConstantMean()
# gplvm.random_effect_mean = gpytorch.means.ZeroMean()
# gplvm.covar_module = gpytorch.kernels.LinearKernel(q + len(X_covars.T))

# likelihood = NBLikelihood(d)

# if torch.cuda.is_available():
#     Y = Y.cuda()
#     gplvm = gplvm.cuda()
#     gplvm.X_covars = gplvm.X_covars.cuda()
#     likelihood = likelihood.cuda()
#     gplvm.X_latent = gplvm.X_latent.cuda()

# batch_size = 100 ## how much does the RAM allow here?
# n_epochs = 80 

# losses = train(gplvm=gplvm, likelihood=likelihood, Y=Y, lr=0.005, epochs=n_epochs, batch_size=batch_size) # TODO: check if you can make this run faster

# # if os.path.exists('latent_sd.pt'):
# # torch.save(losses, 'models/latent_sd_noscale_losses.pt')
# # torch.save(gplvm.X_latent.state_dict(), 'models/latent_sd_scaleexpt_statedict.pt')
# X_latent = ScalyEncoder(q, d + X_covars.shape[1]) 
# X_latent.load_state_dict(torch.load('models/latent_sd_scaleexpt_statedict.pt'))
# X_latent.eval()
# X_latent.cuda()

# #-- plotting --
# # find X_latent means directly
# # z_params = gplvm.X_latent.z_nnet(torch.cat([Y.cuda(), X_covars.cuda()], axis=1))
# z_params = X_latent.z_nnet(torch.cat([Y.cuda(), X_covars.cuda()], axis=1))

# # X_latent_dims =  z_params[..., :gplvm.X_latent.latent_dim]
# X_latent_dims = z_params[..., :X_latent.latent_dim]

# adata.obsm['X_BGPLVM_latent'] = X_latent_dims.detach().cpu().numpy()


# gplvm_bio_metrics, __ = calc_bio_metrics(adata, embed_key = 'X_BGPLVM_latent')
# torch.save(gplvm_batch_metrics, 'models/gplvm_bio_metrics.pt')

# gplvm_batch_metrics = calc_batch_metrics(adata, embed_key = 'X_BGPLVM_latent')
# torch.save(gplvm_batch_metrics, 'models/gplvm_batch_metrics.pt')

# sc.pp.neighbors(adata, n_neighbors=50, use_rep='X_BGPLVM_latent', key_added='BGPLVM')
# sc.tl.umap(adata, neighbors_key='BGPLVM')
# adata.obsm['X_umap_BGPLVM'] = adata.obsm['X_umap'].copy()

# plt.rcParams['figure.figsize'] = [10,10]
# col_obs = ['harmonized_celltype', 'Site']
# sc.pl.embedding(adata, 'X_umap_BGPLVM', color = col_obs, legend_loc='on data', size=5)

