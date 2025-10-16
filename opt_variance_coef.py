import numpy as np

import scipy as sc
from scipy import io
from scipy import optimize
import states_functions as sf

# Functions for the construction and optimisation of the optimal variance of a target observable given a specific POVM
# Requires "states_function" module for some optimised functions
## variance_pc:               returns the variance given a set of coefficients and probability distribution
## variance_coef_state:       returns the variance given a set of coefficients and fixed state
## variance_coef_frevars:     analogue to previous, but construct state from free real variables (specific for optimisation)
## nnz_probs:                 given a fixed state, returns the expected probability distribution with a slight variation (<0.1%) to guarantee there are no zero probabilities
## opt_invmat_state:          given a target observable returns the optimal estimator matrix  for a fixed state
## opt_coef_state:            given a target observable returns the optimal coefficients for a fixed state
## opt_sn_state:              given a target observable returns the optimal shadow norm for a fixed state
## var_state_optimisation:    receives as input the free variables, returns the optimal *variance* (relies on opt_sn_state)
## variance_optimisation:     handles the optimisation of the variance for given target observable, also returns maximal state and optimal coefficients
## fix_coef_var_optimisation: optimisation of variance for fixed set of coefficients (therefore maximisation on state)

def variance_pc (probs, coefs):
    # variance from choice of coefficients and probability distribution
    c2 = np.conj(coefs)*coefs
    return probs@c2 -np.real((probs@coefs))**2

                ##-----##

def variance_coef_state (state, povm_mat, coefs):
    # variance from choice of coefficients and fixed state
    # probability distribution is obtained directly from the state:
    probs = nnz_probs(povm_mat, state)
    
    return variance_pc(probs, coefs)

                ##-----##
    
def variance_coef_freevars (x, povm_mat, basis_mat, coefs):
    # *negative variance* from choice of coefficients and free variables (specific for fix_coef_var_optimisation)
    # (negativity imposed to turn concave function into convex for minimisation)
    
    state = sf.flat_state_from_mat (x, basis_mat) # adapts free variables into a density matrix (in proper subspace)
    probs = nnz_probs(povm_mat, state)            # returns probability ditribution given the reconstructed state
    
    return -np.real(variance_pc(probs, coefs)) # real part imposed to avoid optimisation problems

                ##-----##

def nnz_probs(povm_mat, flat_rho):
    probs = povm_mat @ flat_rho
    eps = 1e-12               # small floor to avoid zero probabilities
    probs = np.clip(probs, eps, None)  # set minimum probability
    probs /= probs.sum()       # renormalize to 1
    return probs
                ##-----##

def opt_invmat_state (povm_mat, probs):
    # optimal estimator matrix for a given fixed state, given its projection on the POVM effects
    # represents a variation of usual Moore-Penrose pseudo-inverse
    dmat = np.diag(np.reciprocal(probs))
    
    return dmat@povm_mat@sc.linalg.inv(povm_mat.T @ dmat @ povm_mat)

                ##-----##

def opt_coef_state (povm_mat, flat_obs, flat_rho):
    # optimal coefficients for the variance (and shadow norm) for a set of probabilities (fixed state)
    probs = nnz_probs(povm_mat, flat_rho)
    lmat = opt_invmat_state(povm_mat, probs)
    
    return lmat@flat_obs

                ##-----##

def opt_sn_state (povm_mat, flat_obs, flat_rho):
    # optimal shadow norm (first term in variance) given an observable and a particular state
    # uses previous functions for calls

    probs = nnz_probs(povm_mat, flat_rho)
    coefs = opt_coef_state(povm_mat, flat_obs, flat_rho)
    dmat = np.diag(probs)
    
    return np.real(np.conj(coefs)@dmat@coefs) # this should always be positive, reality imposed to ignore null imaginary part in function handling

                ##-----##

def var_state_optimisation (x, povm_mat, basis_mat, flat_obs):
    # actual optimisation target function, constructing the state from the free parameters and estimates the variance 
    # using the optimal coefficients
    rho_flat = sf.flat_state_from_mat (x, basis_mat) # adapts free variables into a density matrix (in proper subspace)
    
    return - opt_sn_state(povm_mat, flat_obs, rho_flat) + np.real((rho_flat@flat_obs)**2)

                ##-----##

def variance_optimisation(povm_cm, basis_m ,observable):
    # the real heart of the module, where the magic actually happens
    # takes as input - the POVM coefficient matrix
    #                - the proper basis matrix
    #                - the target observable
    
    d = len(observable)  # this is a matrix, so this is dimension of *vector* space
    D = d**2
    flatobs = sf.flatten_in_basis(observable, basis_m)
    
    x0 = np.ones(D)/(D) # "maximally mixed" initial condition

    res = optimize.minimize(
    var_state_optimisation,
    x0,
    args=(povm_cm, basis_m, flatobs),
    method='Nelder-Mead', # <--- new method
    options={
    'xatol': 1e-5, # tolerance for parameters
    'fatol': 1e-5, # tolerance for function value
    'maxiter': 1000, # max iterations (adjust as needed)
    'disp': False # set True to see convergence messages
    })

    
    xs = res.x
    flat_rhos = sf.flat_state_from_mat(xs, basis_m) # optimal state
    ocs  = opt_coef_state(povm_cm, flatobs, flat_rhos)
    rhos = np.reshape(basis_m@flat_rhos,(d,d))
    var  = -res.fun
    
    return [var, rhos, ocs]

                ##-----##

def fix_coef_var_optimisation(povm_cm, basis_m, coefs):
    # given an already valid set of coefficients, returns the maximal variance for that particular decomposition
    # takes as input - the POVM coefficient matrix
    #                - the proper basis matrix
    #                - the coefficients chosen (these contain information about the target observable)
    
    D = max(basis_m.shape)  # dimension of HS space
    d = int(np.sqrt(D))
    
    x0 = np.ones(D)/(D) # "maximally mixed" initial condition

    res = optimize.minimize(
    variance_coef_freevars,
    x0,
    args=(povm_cm, basis_m, coefs),
    method='Nelder-Mead', # <--- new method
    options={
    'xatol': 1e-5, # tolerance for parameters
    'fatol': 1e-5, # tolerance for function value
    'maxiter': 1000, # max iterations (adjust as needed)
    'disp': False # set True to see convergence messages
    })


    xs = res.x
    flat_rhos = sf.flat_state_from_mat(xs, basis_m) # optimal state
    rhos = np.reshape(basis_m@flat_rhos,(d,d))
    var  = -res.fun
    
    return [var, rhos]
