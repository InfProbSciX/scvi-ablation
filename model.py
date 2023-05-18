
import torch
import gpytorch
import numpy as np
from tqdm import trange

from torch.distributions import Normal, LogNormal
from torch.distributions import kl_divergence
from scvi.distributions import NegativeBinomial

from gpytorch.mlls import VariationalELBO
from gpytorch.constraints import Interval
from gpytorch.models import ApproximateGP
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from gpytorch.mlls.added_loss_term import AddedLossTerm
from gpytorch.means import ConstantMean, LinearMean, ZeroMean
from gpytorch.variational import VariationalStrategy, \
    CholeskyVariationalDistribution
from gpytorch.kernels import ScaleKernel, LinearKernel, RBFKernel, \
    PeriodicKernel

softplus = torch.nn.Softplus()

class GPLVM(ApproximateGP):
    def __init__(self, n, data_dim, latent_dim, covariate_dim,
                 pseudotime_dim=True, n_inducing=60, period_scale=2*np.pi,
                 X_latent=None, X_covars=None,
                 learn_inducing_locations=True):
        self.n = n
        self.q_l = latent_dim
        self.m = n_inducing
        self.q_c = covariate_dim
        self.q_p = pseudotime_dim
        self.batch_shape = torch.Size([data_dim])

        self.learn_inducing_locations = learn_inducing_locations
        self.inducing_inputs = torch.randn(
            n_inducing, latent_dim + pseudotime_dim + covariate_dim)
        if pseudotime_dim:
            self.inducing_inputs[:, 0] = \
                torch.linspace(0, period_scale, n_inducing)

        q_u = CholeskyVariationalDistribution(n_inducing,
                                              batch_shape=self.batch_shape)
        q_f = VariationalStrategy(self, self.inducing_inputs,
                                  q_u, learn_inducing_locations=learn_inducing_locations)
        super(GPLVM, self).__init__(q_f)

        self._init_gp_mean(covariate_dim)
        self._init_gp_covariance(
            data_dim, latent_dim, pseudotime_dim, covariate_dim, period_scale)
        self.X_latent = X_latent
        self.X_covars = X_covars

    def _init_gp_mean(self, covariate_dim):
        self.intercept = ConstantMean()
        if covariate_dim:
            self.random_effect_mean = LinearMean(covariate_dim, bias=False)
        else:
            self.random_effect_mean = ZeroMean()

    def _init_gp_covariance(self, d, q, pseudotime_dim, covariate_dim,
                            period_scale):

        self.pseudotime_dims = list(range(pseudotime_dim))
        if len(self.pseudotime_dims):
            period_length = Interval(period_scale-0.01, period_scale)
            pseudotime_covariance = PeriodicKernel(
                ard_num_dims=len(self.pseudotime_dims),
                active_dims=self.pseudotime_dims,
                period_length_constraint=period_length
            )
        else:
            pseudotime_covariance = None

        self.latent_var_dims = np.arange(pseudotime_dim, pseudotime_dim + q)
        if len(self.latent_var_dims):
            latent_covariance = RBFKernel(
                ard_num_dims=len(self.latent_var_dims),
                active_dims=self.latent_var_dims
            )
        else:
            latent_covariance = None

        max_dim = max(self.latent_var_dims, default=-1)
        max_dim = max(max_dim, max(self.pseudotime_dims, default=-1))
        self.known_var_dims = np.arange(covariate_dim + max_dim, max_dim, -1)
        self.known_var_dims.sort()
        if len(self.known_var_dims):
            random_effect_covariance = LinearKernel(
                ard_num_dims=len(self.known_var_dims),
                active_dims=self.known_var_dims
            )
        else:
            random_effect_covariance = None

        if not random_effect_covariance and not latent_covariance and \
           not pseudotime_covariance:
            raise ValueError('At least one covariance must be specified.')

        if pseudotime_covariance and latent_covariance:
            self.covar_module = pseudotime_covariance * latent_covariance
        elif pseudotime_covariance:
            self.covar_module = pseudotime_covariance
        elif latent_covariance:
            self.covar_module = latent_covariance
        else:
            self.covar_module = random_effect_covariance

        if (pseudotime_covariance or latent_covariance) and \
           random_effect_covariance:
            self.covar_module += random_effect_covariance

        self.covar_module = ScaleKernel(self.covar_module)
        # batch_shape=torch.Size([d])

    def forward(self, X):
        mean_x = self.intercept(X) + \
                 self.random_effect_mean(X[..., self.known_var_dims])
        covar_x = self.covar_module(X)
        dist = MultivariateNormal(mean_x, covar_x)
        return dist


class BatchIdx:
    def __init__(self, n, max_batch_size):
        self.n = n
        self.batch_size = max_batch_size
        self.indices = np.arange(n)
        np.random.shuffle(self.indices)

    def idx(self):
        min_idx = 0
        while True:
            min_idx_incr = min_idx + self.batch_size
            max_idx = min_idx_incr if (min_idx_incr <= self.n) else self.n
            yield self.indices[min_idx:max_idx]
            min_idx = 0 if min_idx_incr >= self.n else min_idx_incr


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
        iterator.set_description(f'L:{np.round(loss.item(), 2)}')
        loss.backward()
        optimizer.step()

    return losses

class LatentVariable(gpytorch.Module):
    pass


class PointLatentVariable(LatentVariable):
    def __init__(self, X_init):
        super().__init__()
        self.register_parameter('X', torch.torch.nn.Parameter(X_init)) # check this code

    def forward(self, batch_index=None, Y=None):
        return self.X[batch_index, :] if batch_index is not None \
               else self.X

class VariationalLatentVariable(LatentVariable):
    def __init__(self, X_init, data_dim):
        n, latent_dim = X_init.shape
        super().__init__()
        
        self.n = n
        self.data_dim = data_dim
        self.latent_dim = latent_dim
        
        self.prior_x = NormalPrior(
            torch.zeros(1, latent_dim),
            torch.ones(1, latent_dim)
        )
        
        # Local variational params per latent point with dimensionality latent_dim
        self.q_mu = torch.nn.Parameter(X_init)
        self.q_log_sigma = torch.nn.Parameter(torch.randn(n, latent_dim))     

        self.register_added_loss_term("x_kl")

    def forward(self, batch_index=None):
        
        if batch_index is None:
            batch_index = np.arange(self.n) 
        
        q_mu_batch = self.q_mu[batch_index, ...]
        q_log_sigma_batch = self.q_log_sigma[batch_index, ...]

        q_x = Normal(q_mu_batch, q_log_sigma_batch.exp())
        
        x_kl = _KL(q_x, self.prior_x, len(batch_index), self.data_dim)        
        self.update_added_loss_term('x_kl', x_kl)
        
        return q_x.rsample()

class ccVariationalLatentVariable(VariationalLatentVariable):
    def __init__(self, X_init, data_dim, cc_init = None):
        super().__init__( X_init, data_dim)
        self.cc_init = cc_init
        self.cc_latent = torch.nn.Parameter(cc_init)
        
    def forward(self, batch_index = None): 
        q_xsample = super().forward(batch_index)
        # cc_kl = (kappa_q)/(q_xsample.shape[0]*self.data_dim) <- vonmises doesn't have an rsample
        return torch.cat([self.cc_latent[batch_index,:], q_xsample], axis = 1)

class NNEncoder(LatentVariable):    
    def __init__(self, n, latent_dim, data_dim, layers):
        super().__init__()
        self.n = n
        self.latent_dim = latent_dim
        self.prior_x = NormalPrior(
            torch.zeros(1, latent_dim),
            torch.ones(1, latent_dim))
        self.data_dim = data_dim

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

    def forward(self, batch_index = None, Y=None):
        h = self.nnet(Y)
        mu = h[..., :self.latent_dim].tanh()*5
        sg = softplus(h[..., self.latent_dim:]) + 1e-6

        q_x = torch.distributions.Normal(mu, sg)

        x_kl = _KL(q_x, self.prior_x, len(mu), self.data_dim)
        self.update_added_loss_term('x_kl', x_kl)
        return q_x.rsample()

class ccNNEncoder(NNEncoder):
    def __init__(self, n, latent_dim, data_dim, layers, cc_init=None):
        super().__init__(n, latent_dim, data_dim, layers)
        self.cc_init = cc_init
        self.cc_latent = torch.nn.Parameter(cc_init)

        
    def forward(self, Y=None, batch_index = None): 
        q_xsample = super().forward(Y=Y)
        print(q_xsample.shape)
        print(torch.cat([self.cc_latent[batch_index,:], q_xsample], axis = 1).shape)
        print(Y.shape)
        print(batch_index.shape)
        # cc_kl = (kappa_q)/(q_xsample.shape[0]*self.data_dim) <- vonmises doesn't have an rsample
        return torch.cat([self.cc_latent[batch_index,:], q_xsample], axis = 1)

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
            self.prior_l = LogNormal(loc=0, scale=1)  

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

    def forward(self, Y=None, X_covars=None, batch_index = None):
        if(X_covars is not None):
            z_params = self.z_nnet(torch.cat([Y, X_covars], axis=1))
        else:
            z_params = self.z_nnet(Y)

        q_x = torch.distributions.Normal(
            z_params[..., :self.latent_dim],
            softplus(z_params[..., self.latent_dim:]) + 1e-4
        )

        ## Adding KL(q|p) loss term 
        x_kl = _KL(q_x, self.prior_x, Y.shape[0], self.input_dim)
        self.update_added_loss_term('x_kl', x_kl)
        

        if(self.learn_scale):
            if(X_covars is not None):
                l_params = self.l_nnet(torch.cat([Y, X_covars], axis=1))
            else:
                l_params = self.l_nnet(Y)
                
            q_l = torch.distributions.LogNormal(
                l_params[..., :1].tanh()*10,
                l_params[..., 1:].sigmoid()*10 + 1e-4
            )
            
            row_sums = torch.sum(Y, dim=1)
            
            empirical_total_mean = torch.mean(row_sums)
            empirical_total_var = torch.std(row_sums)
            self.prior_l = LogNormal(loc = empirical_total_mean, scale = empirical_total_var)

            ## Adding KL(q|p) loss term 
            l_kl = _KL(q_l, self.prior_l, Y.shape[0], self.input_dim)
            self.update_added_loss_term('l_kl', l_kl)
        
        if(self.learn_scale):
            return q_x.rsample(), q_l.rsample()
        return q_x.rsample()

class ccScalyEncoder(ScalyEncoder):
    def __init__(self, latent_dim, input_dim, learn_scale, Y, cc_init=None):
        super().__init__(latent_dim, input_dim, learn_scale, Y )
        self.cc_init = cc_init
        self.cc_latent = torch.nn.Parameter(cc_init)
        
    def forward(self, Y=None, X_covars = None, batch_index = None): 
        if(self.learn_scale):
            q_xsample, q_lsample = super().forward(Y=Y, X_covars = X_covars)
            return torch.cat([self.cc_latent[batch_index,:], q_xsample], axis =1), q_lsample
        q_xsample= super().forward(Y=Y, X_covars = X_covars)
        return torch.cat([self.cc_latent[batch_index,:], q_xsample], axis =1)
    

# class NNEncoder(LatentVariable):    
#     def __init__(self, n, latent_dim, data_dim, layers):
#         super().__init__()
#         self.n = n
#         self.latent_dim = latent_dim
#         self.prior_x = NormalPrior(
#             torch.zeros(1, latent_dim),
#             torch.ones(1, latent_dim))
#         self.data_dim = data_dim
#         self.latent_dim = latent_dim

#         self._init_nnet(layers)
#         self.register_added_loss_term("x_kl")

#         self.jitter = torch.eye(latent_dim).unsqueeze(0)*1e-5

#     def _init_nnet(self, hidden_layers):
#         layers = (self.data_dim,) + hidden_layers + (self.latent_dim*2,)
#         n_layers = len(layers)

#         modules = []; last_layer = n_layers - 1
#         for i in range(last_layer):
#             modules.append(torch.nn.Linear(layers[i], layers[i + 1]))
#             if i < last_layer - 1: modules.append(softplus)

#         self.z_nnet = torch.nn.Sequential(*modules)

#     def forward(self, batch_index=None, Y=None):
#         h = self.z_nnet(Y)
#         mu = h[..., :self.latent_dim].tanh()*5
#         sg = softplus(h[..., self.latent_dim:]) + 1e-6 

#         q_x = torch.distributions.Normal(mu, sg)

#         x_kl = _KL(q_x, self.prior_x, len(mu), self.data_dim)
#         self.update_added_loss_term('x_kl', x_kl)
#         return q_x.rsample()

class _KL(AddedLossTerm):
    def __init__(self, q_x, p_x, n, d):
        self.q_x = q_x
        self.p_x = p_x
        self.n = n
        self.d = d

    def loss(self):
        kl_per_latent_dim = kl_divergence(self.q_x, self.p_x).sum(axis=0)
        kl_per_point = kl_per_latent_dim.sum()/self.n
        return (kl_per_point/self.d)

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

__all__ = ['GPLVM', 'PointLatentVariable', 'NNEncoder', 'BatchIdx'] # 'train']
