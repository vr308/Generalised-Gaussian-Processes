#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
co2 - doubly collapsed hmc

"""
import gpytorch 
import torch
import theano.tensor as tt
import numpy as np
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.kernels import RQKernel, PeriodicKernel, MaternKernel
from gpytorch.distributions import MultivariateNormal
import matplotlib.pyplot as plt
from utils.config import DATASET_DIR, LOG_DIR
import scipy.stats as st
import pymc3 as pm
import pandas as pd

torch.manual_seed(47)
np.random.seed(45)

def load_co2_dataset(year_split):
    
    index_year_dict = {1990: 394, 1995: 454, 2000: 514, 2005: 574, 2010: 634}
    
    df = pd.read_table(str(DATASET_DIR) + '/mauna.txt', names=['year', 'co2'], infer_datetime_format=True, na_values=-99.99, delim_whitespace=True, keep_default_na=False)
    
    # creat a date index for the data - convert properly from the decimal year 
    
    #df.index = pd.date_range(start='1958-01-15', periods=len(df), freq='M')
    
    df.dropna(inplace=True)
    
    first_co2 = df['co2'][0]
    std_co2 = np.std(df['co2'])   
    
    # normalize co2 levels
       
    y = (df['co2'] - first_co2)/std_co2
    t = df['year'] - df['year'][0]
    
    sep_idx = index_year_dict[year_split]
    
    ## testing 5 future years accounting for 60 data points
    
    y_train = y[0:sep_idx].values
    y_test = y[sep_idx:sep_idx+60].values
    t_train = t[0:sep_idx].values[:,None]
    t_test = t[sep_idx:sep_idx+60].values[:,None]
    return y_train, t_train, y_test, t_test, std_co2

class BayesianSparseGPR_HMC(gpytorch.models.ExactGP):
    
   """ The sparse GP class for regression with the doubly 
        collapsed stochastic bound.
        q(u) is implicit 
        theta is sampled using HMC based on pre-specified intervals
   """
      
   def __init__(self, train_x, train_y, likelihood, Z_init):
        
        super(BayesianSparseGPR_HMC, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.inducing_points = Z_init
        self.num_inducing = len(Z_init)  
        self.data_dim = self.train_x.shape[1]                                                                                          
        self.mean_module = ZeroMean()
        
        ## custom co2 kernel 
        
        self.covar_seasonal = ScaleKernel(PeriodicKernel(ard_num_dims=self.data_dim, period_length=1.0)*RBFKernel(ard_num_dims=1))
        self.covar_trend = ScaleKernel(RBFKernel(ard_num_dims=self.data_dim))
        self.covar_medium = ScaleKernel(RQKernel(ard_num_dims=self.data_dim))
        self.covar_noise = ScaleKernel(RBFKernel(ard_num_dims=self.data_dim))

        self.covar_seasonal.base_kernel[0].kernels[0].period_length = 1.0
        self.covar_seasonal.base_kernel[0].kernels[0].raw_period_length.requires_grad = False
     
        self.base_covar_module = self.covar_trend + self.covar_medium + self.covar_seasonal + self.covar_noise
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=Z_init, likelihood=likelihood)

   def forward(self, x): 
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
   def freeze_kernel_hyperparameters(self):
       
        for name, parameter in self.named_hyperparameters():
           if (name not in  ('covar_module.inducing_points', 'mean_module.weights', 'mean_module.bias')):
               parameter.requires_grad = False
                
        
   def sample_optimal_variational_hyper_dist(self, n_samples, input_dim, Z_opt, tune, sampler_params):
              
       with pm.Model() as model_pymc3:
                
            #coef = pm.Normal('coef', 0, 3)
            #c = pm.Normal('c', 0, 3)
            
            #mean = pm.gp.mean.Linear(coeffs=self.mean_module.weights.detach().numpy(), intercept=self.mean_module.bias.detach().numpy())
            # yearly periodic component x long term trend
            
            log_n_per = pm.Normal('log_n_per', mu=0, sd=3)
            log_l_pdecay = pm.Normal('log_l_pdecay', mu=0, sd=0.1)
            log_l_psmooth = pm.Normal('log_l_psmooth', mu=0, sd=1)
            
            n_per = pm.Deterministic("n_per", tt.exp(log_n_per))
            l_pdecay = pm.Deterministic("l_pdecay", tt.exp(log_l_pdecay))
            l_psmooth = pm.Deterministic("l_psmooth", tt.exp(log_l_psmooth))
            cov_seasonal = (
                n_per ** 2 * pm.gp.cov.Periodic(1, period=1, ls=l_psmooth) * pm.gp.cov.ExpQuad(1, l_pdecay)
            )
            # small/medium term irregularities
            log_n_med = pm.Normal('log_n_med', mu=0, sd=3)
            log_l_med = pm.Normal('log_l_med', mu=0, sd=3)
            log_alpha = pm.Normal('log_alpha', mu=0, sd=0.1)
            
            n_med = pm.Deterministic("n_med", tt.exp(log_n_med))
            l_med = pm.Deterministic("l_med", tt.exp(log_l_med))
            alpha = pm.Deterministic("alpha", tt.exp(log_alpha))
            
            cov_medium = n_med ** 2 * pm.gp.cov.RatQuad(1, l_med, alpha)
        
            # long term trend
            log_n_trend = pm.Normal('log_n_trend', mu=0, sd=3)
            log_l_trend = pm.Normal('log_l_trend', mu=0, sd=1)
            
            n_trend = pm.Deterministic("n_trend", tt.exp(log_n_trend))
            l_trend = pm.Deterministic("l_trend", tt.exp(log_l_trend))
            cov_trend = n_trend ** 2 * pm.gp.cov.ExpQuad(1, l_trend)
        
            # noise model
            
            log_n_noise = pm.Normal('log_n_noise', mu=0, sd=3)
            log_l_noise = pm.Normal('log_l_noise', mu=0, sd=1)
            #log_sigma = pm.Normal('log_sigma', mu=0, sd=3)
            
            n_noise = pm.Deterministic("n_noise", tt.exp(log_n_noise))
            l_noise = pm.Deterministic("l_noise", tt.exp(log_l_noise))
            sigma = pm.HalfNormal("sigma",sigma=1)
            cov_noise = n_noise ** 2 * pm.gp.cov.Matern32(1, l_noise)
            
            # The Gaussian process is a sum of these three components
            cov_final = cov_seasonal + cov_medium + cov_trend + cov_noise
            gp = pm.gp.MarginalSparse(cov_func=cov_final, approx='VFE')
            
            # Z_opt is the intermediate inducing points from the optimisation stage
            y_ = gp.marginal_likelihood("y", X=self.train_x.numpy(), Xu=Z_opt, y=self.train_y.numpy(), noise=sigma)
    
            if sampler_params is not None:
                step = pm.NUTS(step_scale = sampler_params['step_scale'])
            else:
                step = pm.NUTS()    
            trace = pm.sample(n_samples, tune=tune, chains=1, step=step, return_inferencedata=False)   
            
       return trace
   
   def update_model_to_hyper(self, elbo, hyper_sample):
       
        ## mean
        #elbo.model.mean_module.weights = torch.nn.Parameter(torch.tensor(hyper_sample['coef'], dtype=torch.float32).reshape(1,1), requires_grad=False)
        #elbo.model.mean_module.bias = torch.nn.Parameter(torch.tensor(hyper_sample['c'], dtype=torch.float32).reshape(1), requires_grad=False)
       
        ## noise 
        elbo.likelihood.noise_covar.noise = hyper_sample['sigma']**2
        
        ## covar_trend
        elbo.model.covar_trend.outputscale = hyper_sample['n_trend']**2
        elbo.model.covar_trend.base_kernel.lengthscale = hyper_sample['l_trend']

        ## covar_seasonal
        elbo.model.covar_seasonal.base_kernel[0].kernels[0].outputscale = hyper_sample['n_per']**2
        elbo.model.covar_seasonal.base_kernel[0].kernels[0].lengthscale = hyper_sample['l_psmooth']
        #elbo.model.covar_seasonal.kernels[0].base_kernel.period_length = hyper_sample['period']
        elbo.model.covar_seasonal.base_kernel[1].kernels[1].lengthscale = hyper_sample['l_pdecay']
        
        ## covar_medium
        elbo.model.covar_medium.outputscale = hyper_sample['n_med']**2
        elbo.model.covar_medium.base_kernel.lengthscale = hyper_sample['l_med']
        elbo.model.covar_medium.base_kernel.alpha = hyper_sample['alpha']

        ## covar_noise
        elbo.model.covar_noise.base_kernel.lengthscale = hyper_sample['l_noise']
        elbo.model.covar_noise.outputscale = hyper_sample['n_noise']**2
           
   def train_model(self, optimizer, max_steps=10000, hmc_scheduler=[200,500,1000,1500]):

        self.train()
        self.likelihood.train()
        elbo = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)
        
        losses = []
        trace_hyper = None
        trace_step_size = []
        trace_perf_time= []
        logged = False
        
        for n_iter in range(max_steps):
            
          optimizer.zero_grad()
          if n_iter < hmc_scheduler[0]: ##  warmstart phase (no sampling has occured, just optimise as normal)
          
              if not logged:
                  print('----------------Warm Start Phase--------------------')
                  logged=True
                  
              output = self(self.train_x)
              loss = -elbo(output, self.train_y)
              
              losses.append(loss.item())
              loss.backward()
              optimizer.step()
        
          else:
              ## Make sure to always freeze hypers before optimising for inducing locations
              self.covar_seasonal.base_kernel[0].kernels[0].period_length = 1.0
              self.freeze_kernel_hyperparameters()
              ### Compute stochastic elbo loss 
              if trace_hyper is not None:
                  loss = 0.0
                  for i in range(len(trace_hyper)):
                      hyper_sample = trace_hyper[i]
                      self.update_model_to_hyper(elbo, hyper_sample)
                      output = self(self.train_x)
                      loss += -elbo(output, self.train_y).sum()/len(trace_hyper)
                  print('Iter %d/%d - Loss: %.3f ' % (n_iter, max_steps, loss.item())),
                  losses.append(loss.item())
                  loss.backward()
                  optimizer.step()
              if n_iter in hmc_scheduler: ## alternate to hmc sampling of hypers
              
                    print('---------------HMC step start------------------------------')
                    print('Iter %d/%d - Loss: %.3f ' % (n_iter, max_steps, loss.item()) + '\n'),
                    #Z_opt = self.inducing_points.detach().cpu().numpy()
                    Z_opt = self.inducing_points.numpy()
                    
                    if n_iter in (hmc_scheduler[0], hmc_scheduler[-1]):
                        num_tune = 200
                        num_samples = 50
                        #sampler_params=None
                    else:
                        num_tune = 25
                        num_samples = 10
                        #prev_step_size = trace_hyper.get_sampler_stats('step_size')[0]
                        #sampler_params = {'step_scale': prev_step_size}
                        #sampler_params = None
                    trace_hyper = self.sample_optimal_variational_hyper_dist(num_samples, self.data_dim, Z_opt, num_tune, sampler_params=None)  
                    print('---------------HMC step finish------------------------------')
                    trace_step_size.append(trace_hyper.get_sampler_stats('step_size')[0])
                    trace_perf_time.append(trace_hyper.get_sampler_stats('perf_counter_diff').sum())       
        return losses, trace_hyper, trace_step_size, trace_perf_time
    
   def train_fixed_model(self):
       
        self.train()
        self.likelihood.train()
        
        trace_step_size = []
        trace_perf_time = []
        
        print('---------------HMC step start------------------------------')
        Z_opt = self.inducing_points.numpy()#[:,None]
                    
        num_tune = 500
        num_samples = 100
        sampler_params=None
                   
        trace_hyper = self.sample_optimal_variational_hyper_dist(num_samples, self.data_dim, Z_opt, num_tune, sampler_params)  
        print('---------------HMC step finish------------------------------')
        trace_step_size.append(trace_hyper.get_sampler_stats('step_size')[0])
        trace_perf_time.append(trace_hyper.get_sampler_stats('perf_counter_diff').sum())  
        
        return trace_hyper, trace_step_size, trace_perf_time
    
               
   def optimal_q_u(self):
       return self(self.covar_module.inducing_points)
   
   def posterior_predictive(self, test_x):

        ''' Returns the posterior predictive multivariate normal '''

        self.eval()
        self.likelihood.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad():
            y_star = self.likelihood(self(test_x))
        return y_star
    
def mixture_posterior_predictive(model, test_x, trace_hyper):
     
      ''' Returns the mixture posterior predictive multivariate normal '''
     
      # Make predictions by feeding model through likelihood
     
      list_of_y_pred_dists = []
      
      for i in range(len(trace_hyper)):
          
         hyper_sample = trace_hyper[i]
         
         ## Training mode for overwriting hypers 
         model.train()
         model.likelihood.train()
         
         ## mean
         #model.mean_module.weights = torch.nn.Parameter(torch.tensor(hyper_sample['coef'], dtype=torch.float32).reshape(1,1), requires_grad=False)
         #model.mean_module.bias = torch.nn.Parameter(torch.tensor(hyper_sample['c'], dtype=torch.float32).reshape(1), requires_grad=False)
         ## noise 
         likelihood.noise_covar.noise = hyper_sample['sigma']
         
         ## covar_trend
         model.covar_trend.outputscale = hyper_sample['n_trend']**2
         model.covar_trend.base_kernel.lengthscale = hyper_sample['l_trend']

         ## covar_seasonal
         model.covar_seasonal.base_kernel[0].kernels[0].outputscale = hyper_sample['n_per']**2
         model.covar_seasonal.base_kernel[0].kernels[0].lengthscale = hyper_sample['l_psmooth']
         #model.covar_seasonal.kernels[0].base_kernel.period_length = hyper_sample['period']
         model.covar_seasonal.base_kernel[1].kernels[1].lengthscale = hyper_sample['l_pdecay']
         
         ## covar_medium
         model.covar_medium.outputscale = hyper_sample['n_med']**2
         model.covar_medium.base_kernel.lengthscale = hyper_sample['l_med']
         model.covar_medium.base_kernel.alpha = hyper_sample['alpha']

         ## covar_noise
         model.covar_noise.base_kernel.lengthscale = hyper_sample['l_noise']
         model.covar_noise.outputscale = hyper_sample['n_noise']**2
       
         with torch.no_grad():
             
              ## Testing mode for computing the posterior predictive 
              model.eval()
              model.likelihood.eval()
              pred = model.likelihood(model(test_x))
              
              try:
                    chol = torch.linalg.cholesky(pred.covariance_matrix + torch.eye(len(test_x))*1e-5)
                    list_of_y_pred_dists.append(pred)
              except RuntimeError:
                   print('Not psd for sample ' + str(i))
     
      return list_of_y_pred_dists
  
def get_posterior_predictive_uncertainty_intervals(sample_means, sample_stds):
      
      # Fixed at 95% CI
      
      n_test = sample_means.shape[-1]
      components = sample_means.shape[0]
      lower_ = []
      upper_ = []
      for i in np.arange(n_test):
            print(i)
            mix_idx = np.random.choice(np.arange(components), size=2000, replace=True)
            mixture_draws = np.array([st.norm.rvs(loc=sample_means[j,i], scale=sample_stds[j,i]) for j in mix_idx])
            lower, upper = st.scoreatpercentile(mixture_draws, per=[2.5,97.5])
            lower_.append(lower)
            upper_.append(upper)
      return np.array(lower_), np.array(upper_)
  
if __name__ == '__main__':
    
    from utils.metrics import rmse, nlpd_mixture, nlpd

    y_train, t_train, y_test, t_test, std_co2 = load_co2_dataset(year_split=2010)
    
    ## Convert to torch friendly format
    y_train = torch.tensor(y_train, dtype=torch.float32)
    t_train = torch.tensor(t_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    t_test = torch.tensor(t_test, dtype=torch.float32)
    
    ####### Initialising model class, likelihood, inducing inputs ##########
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    Z_init = torch.tensor(np.array(t_train)[np.random.randint(0, len(t_train), 480)])

    model = BayesianSparseGPR_HMC(t_train,y_train, likelihood, Z_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    ######## Custom training depending on model class #########
    
#     trace_hyper, step_sizes, perf_time = model.train_fixed_model()
    max_steps = 500
    break_for_hmc = np.concatenate((np.arange(100,500,50), np.array([max_steps-1])))
    #losses, trace_hyper, step_sizes, perf_time = model.train_model(optimizer, max_steps=max_steps, hmc_scheduler=break_for_hmc)
    trace_hyper, step_sizes, perf_time = model.train_fixed_model()

#     ##### Predictions ###########
    
    Y_test_pred_list = mixture_posterior_predictive(model, t_test, trace_hyper[::2]) ### a list of predictive distributions
    y_mix_loc = np.array([np.array(dist.loc.detach()) for dist in Y_test_pred_list])    
    
# #     #### Compute Metrics  ###########
    
     rmse_test = rmse(torch.tensor(np.mean(y_mix_loc, axis=0)), y_test, std_co2)
     nlpd_test = np.round(nlpd_mixture(Y_test_pred_list, y_test, np.array(std_co2)).item(), 4)

# #     print('Test RMSE: ' + str(rmse_test))
#     print('Test NLPD: ' + str(nlpd_test))
    
#     ################################################
    