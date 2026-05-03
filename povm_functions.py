# povm_functions.py
# This module handles everything related to POVM construction and the linear algebra needed to
# express them in a suitable basis. It is problem-agnostic: the functions here work for any POVM
# on any number of qubits. A student adapting this code for a new problem will likely only need
# to add a new function analogous to pauli_povm_single that defines their POVM of choice;
# tensor_same and povm_coef_matrix can then be used unchanged.

import jax.numpy as jnp

# additional useful functions for POVm generation and handling
## tensor_same:       returns tensor product of the same object N times iteratively - both single ob servables or full POVMs
## povm_coef_matrix:  returns the coefficient matrix in the proper basis (and the basis matrix) of an input POVM
## pauli_povm_single: returns POVM out of Pauli projectors for a single qubit (both plane or full POVMS)


def tensor_same(povm, N):
    # takes the iterative tensor of the *same* object N times
    # can be used both for observables and POVMs (as long as they're arrays of objects)
    # check to avoid mistakes, with cheeky remark
    # Computes povm^{⊗N} by repeated jnp.kron. The first step gives povm⊗povm; each subsequent
    # step prepends another copy of povm (so qubit 1 is in the outermost position of the result).
    # Works for both single matrices (observables) and stacked arrays of matrices (POVMs), as long
    # as the array shape is compatible with jnp.kron. For N=1 the input is returned unchanged.
    N = int(N)
    if N == 1:
        # print('Single POVM, why did you even call this function in the first place?')
        return povm
    elif (N < 1):
        raise ValueError('Invalid value for repetition of tensor product')
    else:
        new_povm = jnp.kron(povm, povm)  # first step, common for any input
        if N > 2:                        # repetition for higher dimensions
            for _ in range(N-2):
                new_povm = jnp.kron(povm, new_povm)
        return new_povm

                ##-----##

def povm_coef_matrix(povm):
    # returns the coefficient matrix expressed in the proper basis of the subspace spanned by the effects
    # also returns the same basis (to also express states and observables in the same basis)
    # (canonical estimator matrix can be easily achieved as Moore-Penrose pseudo-inverse)
    # The POVM effects are vectorised (each d×d matrix becomes a length-D=d^2 row vector) and
    # stacked into an (n, D) matrix. A compact SVD identifies the subspace of dimension nz <= D
    # actually spanned by the effects (nz < D happens e.g. for plane POVMs). The basis matrix bm
    # has shape (D, nz) with orthonormal columns spanning that subspace; pcm = povm_mat @ bm has
    # shape (n, nz) and encodes the POVM in this compressed basis. All downstream optimisation
    # works in this compressed space, which avoids redundant degrees of freedom.
    # To use a different POVM, just pass it here — the rest of the pipeline is unchanged.
    dims = povm.shape
    n = max(dims)
    d = min(dims) # dimension of *vector space* - there should be two of these
    D = d**2      # dimesnion of full *HS* space

    # Use conjugated effect vectors so probabilities are inner products:
    # p_i = Tr(E_i rho) = <vec(E_i), vec(rho)> = conj(vec(E_i)) @ vec(rho)
    can_coef_matrix = jnp.conj(povm.reshape((n, D)))

    # SVD of canonical matrix, to determine the dimension of subspace
    # jnp.linalg.svd returns U, S, Vh like SciPy (full_matrices=False gives compact SVD)
    U, S, Vh = jnp.linalg.svd(can_coef_matrix, full_matrices=False)

    # counting non-zero singular values, corresponding to the dimension
    # of the subspace spanned by the POVM (use small tolerance)
    nz = jnp.sum(jnp.where(jnp.round(S, 10) > 0, 1, 0))
    nz = int(nz)

    # selection of the basis matrix as the set of valid eigenstates from V matrix
    if nz < D:
        bm = Vh[:nz].T  # just to exclude floating point errors
    else:
        bm = jnp.eye(D, dtype=povm.dtype)
    # basis matrix now collects a valid orthonormal basis as its *columns*

    pcm = can_coef_matrix @ bm
    return pcm, bm

                ##-----##

def pauli_povm_single(*args):
    # returns the usual POVM of projectors of Pauli eigenstates
    # since I've already written it multiple times, no reason to do the fancy-schwanzy generation
    # The six effects are the rank-1 projectors onto the ±1 eigenstates of X, Y, Z stored as a
    # (6, 2, 2) array. With no arguments (or one), the full 6-element POVM is returned normalised
    # by 1/3 so that sum_i E_i = I. With two string/int arguments (e.g. 'X','Z'), the four effects
    # for those two Pauli axes are selected and normalised by 1/2.
    # For a new problem: write an analogous function that returns your POVM as an (n, d, d) array
    # and pass it to povm_coef_matrix — no other changes are needed in the rest of the pipeline.
    keyword_to_index = {
        "X" : 0,
        "Y" : 1,
        "Z" : 2,
    }

    pauli_povm = jnp.array([
                            [[0.5,     0.5],[     0.5,  0.5]], # X eigensates
                            [[0.5,    -0.5],[    -0.5,  0.5]],
                            [[0.5,  0.5*1j],[ -0.5*1j,  0.5]], # Y eigenstates
                            [[0.5, -0.5*1j],  [0.5*1j,  0.5]],
                            [[  1,       0],[       0,    0]], # Z eigenstates
                            [[  0,       0],[       0,    1]],
                          ], dtype=jnp.complex128)

    if len(args) < 2:
        return pauli_povm/3 # anything with less than two values will simply return the full POVM
                            # (with proper normalisation to guarantee it sums to identity)
    elif len(args) == 2:

        # selection of corresponding indices in case text is input
        inds = []
        for i in range(2):
            if isinstance(args[i], str):
                try:
                    ind = keyword_to_index[args[i]]
                except KeyError:
                    raise ValueError(f"Unknown keyword: {args[0]}")
            elif isinstance(args[i], int):
                ind = int(args[i])
            else:
                raise TypeError("Input must be a string keyword or an integer index")

            # appended one at the time (to avoid awkward reshaping)
            inds.append(2*ind)
            inds.append(2*ind+1)

        return pauli_povm[jnp.array(inds)]/2 # proper normalisation
    else:
        raise TypeError("Please indicate either the two indices for a plane POVM or nothing at all")
