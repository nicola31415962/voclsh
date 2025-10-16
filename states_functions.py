import numpy as np

# additional useful functions for state handling
## qubit:             generates single-qubit projector
## flatten_in_basis:  flattens any observable in the subspace spanned by the basis matrix
## flat_state_from_mat:  returns (flattened) state from set of real free variables 

def qubit (theta, phi):
    # single qubit projector definition from angle parameters
    vec = np.array([np.cos(theta/2), np.exp(1j*phi)*np.sin(theta/2)])
    return np.round(np.outer(vec,np.conj(vec)),10)

                ##-----##

def flatten_in_basis(obs,bm):
    # returns flattened versions of observable (even density matrices) 
    return np.conj(bm.T)@(obs.flatten())
                   
                ##-----##

def flat_state_from_mat (x, basis_mat):
    # generates a (flattened) quantum states from an array of free parameters interpreted as the
    # real and imaginary part of a trinagular matrix, used to generate the state
    
    D = int(max(basis_mat.shape))  # corresponds to HS dimension
    dd = int(min(basis_mat.shape)) # dimension of subspace
    d = int(np.sqrt(D))            # dimension of *vector* space (to reshape matrix)

    # check of consistent dimensions
    if len(x) != D:
        raise ValueError('dimension of input variables not consistent with basis expressed')
    
    # to easily build the triangular matrix, inputs are reshaped in a square matrix
    xmat = np.reshape(x,(d,d)) 
    re = np.tril(xmat) 
    im = np.tril(xmat.T,k=-1) # this selects all elements in *upper* triangular matrix of xmat
    T = re+1j*im              # triangular matrix
    H = T@np.conj(T.T)        # positive semidefinite version by def.

    if dd < d:              # dimension of subspace smaller than HS space
        # to ensure the matrix is in the relevant subspace, it is projected onto the basis and
        # then expanded back into the full HS space for correct normalisation
        H = np.reshape(basis_mat@flatten_in_basis(H,basis_mat),(d,d))

    # reshaping necessary for proper trace extraction
    rho = H / np.trace(H)                        # normalisation
    
    return  flatten_in_basis(rho, basis_mat) # final projection into proper subspace and flattening