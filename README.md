# Generalised-Gaussian-Processes

Fully Bayesian Inference in GPs - learning hyperparameter distributions with different likelihoods and inference techniques. This repo is a work in progress. 

Likelihoods
-------------

1) Gaussian (Regression)
2) Bernoulli Probit (Classification)
3) SoftMax (Multi-Class)
4) Log Cox Posisson (Regression)

Models
-----------------

1) SGPR (Titsias, 2009)
2) SVGP (Hensman, 2013 / Hensman, 2015)
3) BayesianSVGP (new)
4) BayesianSGPR_HMC (new)

Inference methods
-----------------

1) Hamiltonian Monte Carlo (pymc3)
2) Stochastic Variational Inference (SVI) (gpytorch/pytorch)

Future work (Summer, 2021)
-------------------
3) Elliptical Slice Sampling (pymc3)
4) Dynamic Nested Sampling (pymc3)

Code Layout
-----------------

Please set the working directory to the parent folder which containst the following sub-folders

``` utils/ ``` - Data loading and visualisation utilities.

```experiments/``` - Scripts for running experiments and generating plots.
 
```models/ ``` - Classes and methods which encapsulate inference.

```results/``` - Tables, logging and directory for plots. 
