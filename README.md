# Generalised-Gaussian-Processes

Fully Bayesian Inference in GPs - learning hyperparameter distributions with different likelihoods and inference techniques. This repo is a work in progress. 

Reference (AABI 2019): https://arxiv.org/abs/1912.13440

Likelihoods
-------------

1) Gaussian (Regression)
2) Bernoulli Probit (Classification)
3) SoftMax (Multi-Class)
4) Log Cox Posisson (Regression)


Inference methods
-----------------

1) Hamiltonian Monte Carlo (pymc3)
2) Variational Inference (ADVI) (pymc3)
3) Stochastic Variational Inference (SVI) (gpytorch/pytorch)
4) Elliptical Slice Sampling (pymc3)
5) Dynamic Nested Sampling (pymc3)

Code Layout
-----------------

Please set the working directory to the parent folder which containst the following sub-folders

``` utils/ ``` - Data loading utilities, visualisation utilities.

```experiments/``` - Scripts for running experiments and generating plots.
 
```models/ ``` - Classes and methods which encapsulate inference.

```results/``` - Tables, logging and directory for plots. 
