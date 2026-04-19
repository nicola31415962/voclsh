# opt_variance_coef.py
# Changes:
# - Replaced SciPy/Nelder–Mead with a tiny Adam optimizer (in-file) to stay JAX-native.

import jax
import jax.numpy as jnp
import jax.scipy as jsp

import states_functions as sf  # see note below on import names

# NOTE: If you place this file as jax/opt_variance_coef.py alongside
# jax/states_functions.py, import as:
#   import jax.states_functions as sf
# Adjust to your folder naming (e.g., from . import states_functions as sf).

# default Adam settings (tunable from here)
ADAM_STEPS_MAIN = 200
ADAM_LR_MAIN = 2e-2
ADAM_STEPS_FIX = 200
ADAM_LR_FIX = 2e-2

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

def variance_pc(probs, coefs):
    # variance from choice of coefficients and probability distribution
    c2 = jnp.conj(coefs)*coefs
    return probs@c2 - jnp.real((probs@coefs))**2

                ##-----##

def variance_coef_state(state, povm_mat, coefs):
    # variance from choice of coefficients and fixed state
    # probability distribution is obtained directly from the state:
    probs = nnz_probs(povm_mat, state)
    return variance_pc(probs, coefs)

                ##-----##
    
def variance_coef_freevars(x, povm_mat, basis_mat, coefs):
    # *negative variance* from choice of coefficients and free variables (specific for fix_coef_var_optimisation)
    # (negativity imposed to turn concave function into convex for minimisation)
    state = sf.flat_state_from_mat(x, basis_mat) # adapts free variables into a density matrix (in proper subspace)
    probs = nnz_probs(povm_mat, state)           # returns probability distribution given the reconstructed state
    return -jnp.real(variance_pc(probs, coefs))   # real part imposed to avoid optimisation problems

                ##-----##

def nnz_probs(povm_mat, flat_rho):
    probs = povm_mat @ flat_rho
    # raise the floor; 1e-12 is too small for the inverse below
    eps = 1e-6
    probs = jnp.clip(probs.real, eps, None)
    probs = probs / jnp.sum(probs)
    return probs


                ##-----##

def opt_invmat_state(povm_mat, probs):
    povm_mat = jnp.asarray(povm_mat, dtype=jnp.complex128)
    probs    = jnp.asarray(probs,    dtype=jnp.float64)

    # Clamp probabilities to avoid huge weights
    eps = 1e-6
    probs = jnp.clip(probs, eps, None)

    inv_d = jnp.diag(1.0 / probs)

    # Complex BLUE map for constraints P^T c = conj(flat_obs)
    A = povm_mat.T @ inv_d @ jnp.conj(povm_mat)

    # Scale-aware ridge: lambda is a small fraction of ||A||
    # cheap norm
    scale = jnp.linalg.norm(A, ord=jnp.inf)  # max row-sum, ~max scale in A
    # if A is almost zero, fall back to 1.0 to avoid NaNs
    scale = jnp.where(scale > 0, scale, 1.0)

    eps_reg = 1e-8  # dimensionless, relative to scale
    lam = eps_reg * scale

    A_reg = A + lam * jnp.eye(A.shape[0], dtype=A.dtype)

    I = jnp.eye(A.shape[0], dtype=A.dtype)
    lmat = inv_d @ jnp.conj(povm_mat) @ jnp.linalg.solve(A_reg, I)
    return lmat

                ##-----##

def opt_coef_state(povm_mat, flat_obs, flat_rho):
    # optimal coefficients for the variance (and shadow norm) for a set of probabilities (fixed state)
    probs = nnz_probs(povm_mat, flat_rho)
    lmat  = opt_invmat_state(povm_mat, probs)
    return lmat @ jnp.conj(flat_obs)

                ##-----##

def opt_sn_state(povm_mat, flat_obs, flat_rho):
    # optimal shadow norm (first term in variance) given an observable and a particular state
    # uses previous functions for calls
    probs = nnz_probs(povm_mat, flat_rho)
    coefs = opt_coef_state(povm_mat, flat_obs, flat_rho)
    # Equivalent to coefs^\dagger diag(probs) coefs, in explicit real arithmetic
    # to avoid complex->real casts during reverse-mode autodiff.
    return jnp.sum(probs * (jnp.real(coefs) ** 2 + jnp.imag(coefs) ** 2))

                ##-----##

def var_state_optimisation(x, povm_mat, basis_mat, flat_obs):
    # actual optimisation target function, constructing the state from the free parameters and estimates the variance 
    # using the optimal coefficients
    rho_flat = sf.flat_state_from_mat(x, basis_mat) # adapts free variables into a density matrix (in proper subspace)
    expval = jnp.real(rho_flat @ jnp.conj(flat_obs))
    return - opt_sn_state(povm_mat, flat_obs, rho_flat) + expval ** 2

                ##-----##
# -------- JAX: lightweight Adam to replace SciPy --------

def _adam_minimize(fun, x0, steps=500, lr=2e-2, beta1=0.9, beta2=0.999, eps=1e-8): #default: steps=2000, lr=5e-3; seems stable but risky: steps=100, lr=1e-2
    valgrad = jax.value_and_grad(fun)
    # ensure everything is real (float64) to avoid complex promotions
    x0 = jnp.asarray(x0, dtype=jnp.float64)
    m  = jnp.zeros_like(x0)
    v  = jnp.zeros_like(x0)

    def step(carry, t):
        x, m, v = carry
        val, g = valgrad(x)
        g = jnp.real(g)  # discard any tiny imag parts from complex pathways

        # cast scalars to x's dtype to avoid dtype promotion
        lr_t    = jnp.asarray(lr,    dtype=x.dtype)
        beta1_t = jnp.asarray(beta1, dtype=x.dtype)
        beta2_t = jnp.asarray(beta2, dtype=x.dtype)
        eps_t   = jnp.asarray(eps,   dtype=x.dtype)

        m = beta1_t * m + (1.0 - beta1_t) * g
        v = beta2_t * v + (1.0 - beta2_t) * (g * g)
        t1 = t + 1
        mhat = m / (1.0 - beta1_t ** t1)
        vhat = v / (1.0 - beta2_t ** t1)
        x = x - lr_t * mhat / (jnp.sqrt(vhat) + eps_t)
        return (x, m, v), val

    (x_final, _, _), vals = jax.lax.scan(step, (x0, m, v), jnp.arange(steps))
    return x_final, vals[-1]

# --------------------------------------------------------------------

def variance_optimisation(povm_cm, basis_m, observable):
    # the real heart of the module, where the magic actually happens
    # takes as input - the POVM coefficient matrix
    #                - the proper basis matrix
    #                - the target observable
    
    d = len(observable)  # this is a matrix, so this is dimension of *vector* space
    D = d**2
    flatobs = sf.flatten_in_basis(observable, basis_m)
    
    x0 = jnp.ones((D,), dtype=jnp.float64) / D


    # JAX: replace SciPy minimize with Adam on the same target
    loss = lambda x: var_state_optimisation(x, povm_cm, basis_m, flatobs)
    loss = jax.jit(loss)
    xs, neg_var = _adam_minimize(loss, x0, steps=ADAM_STEPS_MAIN, lr=ADAM_LR_MAIN)

    flat_rhos = sf.flat_state_from_mat(xs, basis_m) # optimal state
    ocs  = opt_coef_state(povm_cm, flatobs, flat_rhos)
    rhos = (basis_m @ flat_rhos).reshape((d, d))
    var  = -neg_var
    
    return [jnp.real(var), rhos, ocs]

                ##-----##

def fix_coef_var_optimisation(povm_cm, basis_m, coefs):
    # given an already valid set of coefficients, returns the maximal variance for that particular decomposition
    # takes as input - the POVM coefficient matrix
    #                - the proper basis matrix
    #                - the coefficients chosen (these contain information about the target observable)
    
    D = max(basis_m.shape)  # dimension of HS space
    d = int(jnp.sqrt(D))
    
    x0 = jnp.ones((D,), dtype=jnp.float64) / D

    def loss(x):
        return jnp.real(variance_coef_freevars(x, povm_cm, basis_m, coefs))

    loss = jax.jit(loss)
    xs, neg_var = _adam_minimize(loss, x0, steps=ADAM_STEPS_FIX, lr=ADAM_LR_FIX)

    flat_rhos = sf.flat_state_from_mat(xs, basis_m) # optimal state
    rhos = (basis_m @ flat_rhos).reshape((d, d))
    var  = -neg_var
    
    return [jnp.real(var), rhos]
