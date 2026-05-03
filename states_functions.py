# JAX PORT of states_functions.py
# This module provides utilities for constructing and manipulating quantum states in the context
# of the optimisation. It is problem-agnostic in the sense that the functions work for any
# Hilbert space dimension, but the choice of STATE_CONSTRAINT does encode a physical assumption
# about which states to consider (all states vs. product pure states). A student adapting this
# code should not need to change anything here unless they want to add a new state parameterisation
# (e.g. states with fixed purity, or states in a particular symmetry sector).

import os

import jax
import jax.numpy as jnp

# additional useful functions for state handling
## qubit:             generates single-qubit projector
## flatten_in_basis:  flattens any observable in the subspace spanned by the basis matrix
## flat_state_from_mat:  returns (flattened) state from set of real free variables

# STATE_CONSTRAINT selects which set of states the optimisation searches over.
# "full"         — all density matrices (default); the most general and physically correct choice.
# "product_pure" — tensor products of pure single-qubit states; a restricted ansatz that is faster
#                  to optimise but will miss entangled worst-case states.
# Set via environment variable before importing this module, e.g.:
#   os.environ["STATE_CONSTRAINT"] = "product_pure"
# or via the USE_PRODUCT_PURE flag in the experiment script (plane_proj_XZ.py).
STATE_CONSTRAINT = os.getenv("STATE_CONSTRAINT", "full")


def qubit(theta, phi):
    # single qubit projector definition from angle parameters
    # Returns the rank-1 density matrix |psi><psi| for the Bloch sphere point (theta, phi),
    # where theta is the polar angle (0 = north pole = |0>) and phi is the azimuthal angle.
    # No rounding is applied (unlike the original NumPy version) to keep gradients clean under
    # JAX autodiff — rounding introduces discontinuities that can confuse the optimiser.
    vec = jnp.array([jnp.cos(theta/2), jnp.exp(1j*phi)*jnp.sin(theta/2)])
    # original rounds to 10 decimals; we avoid rounding to stay JIT/grad-friendly
    return jnp.outer(vec, jnp.conj(vec))

                ##-----##

# Helper functions used exclusively in product_pure mode (see flat_state_from_mat below).
# _sigmoid wraps jax.nn.sigmoid to map unconstrained real parameters in R to (0,1), which
# is then scaled to the appropriate angular range (theta in (0,pi), phi in (0,2pi)).
# _ket_qubit constructs a single-qubit ket |psi_i> from angles (theta_i, phi_i).
# _kron_all_vec iteratively builds the tensor product ket psi_1 ⊗ ... ⊗ psi_N from a list of kets.
def _sigmoid(z):
    return jax.nn.sigmoid(z)


def _ket_qubit(theta, phi):
    return jnp.array([jnp.cos(theta/2), jnp.exp(1j*phi) * jnp.sin(theta/2)])


def _kron_all_vec(vecs):
    out = vecs[0]
    for v in vecs[1:]:
        out = jnp.kron(out, v)
    return out

                ##-----##

def flatten_in_basis(obs, bm):
    # returns flattened versions of observable (even density matrices)
    # Projects obs (a d×d matrix) into the POVM subspace defined by basis matrix bm and returns
    # the coordinate vector of length nz (the subspace dimension). The projection is
    # bm^dagger @ vec(obs), where vec(obs) stacks the columns of obs into a length-D vector.
    # This is used both to express target observables and to flatten density matrices before
    # passing them to the optimisation routines in opt_variance_coef.py.
    return jnp.conj(bm.T) @ (obs.reshape(-1))

                ##-----##

def flat_state_from_mat(x, basis_mat):
    # generates a (flattened) quantum state from an array of free parameters interpreted as the
    # real and imaginary part of a triangular matrix, used to generate the state
    # Maps a real parameter vector x of length D=d^2 to a valid density matrix, then returns its
    # flattened projection onto the POVM subspace (length nz). Two parameterisations are available:
    #
    # "full" mode (default):
    #   x is reshaped into a d×d matrix; the lower triangle gives the real part of a triangular
    #   matrix T and the strict upper triangle gives the imaginary part of its lower triangle.
    #   rho = T T^dagger / Tr(T T^dagger) is then always positive semidefinite and normalised.
    #   This is the Cholesky-like parameterisation standard in quantum tomography.
    #
    # "product_pure" mode:
    #   Only the first 2N entries of x are used (N = log2(d) qubits). x[:N] and x[N:2N] are
    #   mapped via sigmoid to angles theta_i in (0,pi) and phi_i in (0,2pi); the state is the
    #   product pure state ⊗_i |psi_i><psi_i|. This restricts the search to a 2N-dimensional
    #   manifold instead of d^2-dimensional, which speeds up convergence but will miss entangled
    #   worst-case states — use only when you have reason to believe the worst case is separable.
    #
    # In both modes, if the POVM subspace has dimension nz < D the constructed rho is projected
    # onto the subspace and renormalised before flattening, keeping it in the physically relevant
    # subspace throughout the optimisation.

    import numpy as onp  # ensure plain numpy available

    D  = int(basis_mat.shape[0])   # corresponds to HS dimension
    dd = int(basis_mat.shape[1])   # dimension of subspace
    d  = int(onp.sqrt(D))          # dimension of *vector* space (to reshape matrix)

    # check of consistent dimensions
    if x.size != D:
        return jnp.nan * jnp.ones((D,), dtype=basis_mat.dtype)

    state_constraint = STATE_CONSTRAINT.lower()

    if state_constraint == "full":
        # to easily build the triangular matrix, inputs are reshaped in a square matrix
        xmat = x.reshape((d, d))
        re = jnp.tril(xmat)
        im = jnp.tril(xmat.T, k=-1)  # selects all elements in *upper* triangular matrix of xmat
        T = re + 1j*im               # triangular matrix
        H = T @ jnp.conj(T.T)        # positive semidefinite version by def.

        if dd < d:                   # dimension of subspace smaller than HS space
            # ensure the matrix is in the relevant subspace, project onto the basis and
            # expand back into the full HS space for correct normalisation
            H = (basis_mat @ flatten_in_basis(H, basis_mat)).reshape((d, d))

        rho = H / jnp.trace(H)       # normalisation
        return flatten_in_basis(rho, basis_mat)  # final projection into proper subspace and flattening

    if state_constraint == "product_pure":
        N_float = onp.log2(d)
        if int(N_float) != N_float:
            raise ValueError("Product-pure mode requires d to be a power of 2.")
        N = int(N_float)

        a = x[:N]
        b = x[N:2*N]
        theta = jnp.pi * _sigmoid(a)
        phi = 2 * jnp.pi * _sigmoid(b)

        kets = [_ket_qubit(theta[i], phi[i]) for i in range(N)]
        psi = _kron_all_vec(kets)
        rho = jnp.outer(psi, jnp.conj(psi))

        if dd < d:
            rho = (basis_mat @ flatten_in_basis(rho, basis_mat)).reshape((d, d))
            rho = rho / jnp.trace(rho)

        return flatten_in_basis(rho, basis_mat)

    raise ValueError(f"Unsupported STATE_CONSTRAINT '{STATE_CONSTRAINT}'; use 'full' or 'product_pure'.")


if __name__ == "__main__":
    import numpy as onp

    N = 2
    d = 2 ** N
    D = d ** 2
    basis_mat = jnp.eye(D, dtype=jnp.complex128)
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (D,))

    for mode in ("full", "product_pure"):
        os.environ["STATE_CONSTRAINT"] = mode
        STATE_CONSTRAINT = mode
        flat_rho = flat_state_from_mat(x, basis_mat)
        rho = flat_rho.reshape((d, d))
        tr = jnp.trace(rho)
        eigs = jnp.linalg.eigvalsh(rho)
        assert bool(jnp.allclose(tr, 1.0, atol=1e-5))
        assert bool(jnp.all(eigs >= -1e-6))
        assert bool(jnp.allclose(rho, jnp.conj(rho.T)))

    print("ok")
