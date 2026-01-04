# JAX PORT of states_functions.py

import jax
import jax.numpy as jnp

# additional useful functions for state handling
## qubit:             generates single-qubit projector
## flatten_in_basis:  flattens any observable in the subspace spanned by the basis matrix
## flat_state_from_mat:  returns (flattened) state from set of real free variables 

def qubit(theta, phi):
    # single qubit projector definition from angle parameters
    vec = jnp.array([jnp.cos(theta/2), jnp.exp(1j*phi)*jnp.sin(theta/2)])
    # original rounds to 10 decimals; we avoid rounding to stay JIT/grad-friendly
    return jnp.outer(vec, jnp.conj(vec))

                ##-----##

def flatten_in_basis(obs, bm):
    # returns flattened versions of observable (even density matrices) 
    return jnp.conj(bm.T) @ (obs.reshape(-1))
                   
                ##-----##

def flat_state_from_mat(x, basis_mat):
    # generates a (flattened) quantum state from an array of free parameters interpreted as the
    # real and imaginary part of a triangular matrix, used to generate the state

    import numpy as onp  # ensure plain numpy available

    D  = int(basis_mat.shape[0])   # corresponds to HS dimension
    dd = int(basis_mat.shape[1])   # dimension of subspace
    d  = int(onp.sqrt(D))          # dimension of *vector* space (to reshape matrix)

    # check of consistent dimensions
    if x.size != D:
        return jnp.nan * jnp.ones((D,), dtype=basis_mat.dtype)

    # to easily build the triangular matrix, inputs are reshaped in a square matrix
    xmat = x.reshape((d, d))
    re = jnp.tril(xmat)
    im = jnp.tril(xmat.T, k=-1)  # this selects all elements in *upper* triangular matrix of xmat
    T = re + 1j*im               # triangular matrix
    H = T @ jnp.conj(T.T)        # positive semidefinite version by def.

    if dd < d:                   # dimension of subspace smaller than HS space
        # to ensure the matrix is in the relevant subspace, it is projected onto the basis and
        # then expanded back into the full HS space for correct normalisation
        H = (basis_mat @ flatten_in_basis(H, basis_mat)).reshape((d, d))

    rho = H / jnp.trace(H)       # normalisation
    return flatten_in_basis(rho, basis_mat)  # final projection into proper subspace and flattening

