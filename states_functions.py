# JAX PORT of states_functions.py

import os

import jax
import jax.numpy as jnp

# additional useful functions for state handling
## qubit:             generates single-qubit projector
## flatten_in_basis:  flattens any observable in the subspace spanned by the basis matrix
## flat_state_from_mat:  returns (flattened) state from set of real free variables

STATE_CONSTRAINT = os.getenv("STATE_CONSTRAINT", "full")


def qubit(theta, phi):
    # single qubit projector definition from angle parameters
    vec = jnp.array([jnp.cos(theta/2), jnp.exp(1j*phi)*jnp.sin(theta/2)])
    # original rounds to 10 decimals; we avoid rounding to stay JIT/grad-friendly
    return jnp.outer(vec, jnp.conj(vec))

                ##-----##

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
