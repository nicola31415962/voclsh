================================================================================
README — Optimal Variance Estimation for POVMs
================================================================================

What this is
------------
This code finds the optimal estimator (and its worst-case variance) for a target
observable measured via a given POVM. The question being answered is: given that
you are performing a fixed set of measurements (a POVM), what is the best linear
estimator for some observable O, and how bad can the variance get over all possible
input states?

The optimisation is over two things simultaneously:
  1. The coefficient vector c (which defines the estimator: O_est = sum_i c_i * outcome_i)
  2. The input state rho (to find the worst-case state for that estimator)

The result is the maximum variance achievable by the *optimal* estimator — i.e.
the tightest upper bound on how badly any linear estimator can do for this POVM
and observable.

The implementation is JAX-native (uses jit, vmap, lax.scan) and runs on CPU or GPU
without code changes. Optimisation is done with a built-in Adam gradient descent.


Dependencies
------------
  jax, jaxlib
  numpy
  matplotlib

See requirements.txt for pinned versions. JAX float64 support must be enabled
(this is done automatically at the top of each experiment script via
jax.config.update("jax_enable_x64", True)).


Data flow overview
------------------
The minimal pipeline from raw inputs to a result is:

  (1) your POVM (n effects, each a d×d matrix)
        --> pf.povm_coef_matrix(povm)
        --> pcm : (n, nz) complex   POVM coefficient matrix in the subspace basis
            bm  : (D, nz) complex   orthonormal basis of the subspace (D = d^2)

  (2) your observable (d×d Hermitian matrix) + pcm, bm from (1)
        --> ocf.variance_optimisation(pcm, bm, observable)
        --> [variance, rho*, coefs*]
            variance : scalar float    worst-case variance of the optimal estimator
            rho*     : (d, d) complex  worst-case state
            coefs*   : (n,) complex    optimal estimator coefficients at rho*

For benchmarking against the canonical estimator:
  can_em = jnp.linalg.pinv(pcm)                               # (nz, n)
  cc = can_em.T @ conj(sf.flatten_in_basis(obs, bm))          # (n,) canonical coefs
  ocf.fix_coef_var_optimisation(pcm, bm, cc)
        --> [variance, rho*]   worst-case variance for those fixed canonical coefs


File structure
--------------
There are four main Python files. Two roles: core engine (do not modify) and
experiment script (copy and adapt for a new problem).

  CORE ENGINE — problem-agnostic, do not modify for a new problem:

  states_functions.py
      Utilities for constructing and manipulating quantum states. The key function
      is flat_state_from_mat(x, basis_mat), which maps a real parameter vector x
      (length D = d^2) into a valid density matrix projected onto the POVM subspace
      and returns it as a flattened (nz,) complex vector. Two modes are available,
      selected via the STATE_CONSTRAINT environment variable:
        - "full": all density matrices (Cholesky-like parameterisation, default)
        - "product_pure": tensor products of pure single-qubit states (faster,
          but will miss entangled worst-case states)
      Also contains qubit(theta, phi) which returns a single-qubit (2,2) projector,
      and flatten_in_basis(obs, bm) which projects any (d,d) matrix into the POVM
      subspace and returns its (nz,) coordinate vector.

  povm_functions.py
      Utilities for POVM construction and linear algebra. The key function is
      povm_coef_matrix(povm), which takes a POVM as an (n, d, d) array and returns
      pcm (n, nz) and bm (D, nz) — the only POVM-derived objects needed downstream.
      nz <= D is the dimension of the subspace spanned by the effects (nz < D for
      informationally incomplete POVMs like plane POVMs). The canonical estimator
      matrix is just jnp.linalg.pinv(pcm), shape (nz, n).
      Also contains tensor_same(obj, N) which takes any matrix or POVM array and
      returns its N-fold tensor product — used to build N-qubit POVMs and observables
      from single-qubit ones.

  opt_variance_coef.py
      The optimisation engine. The two functions a user will call are:
        - variance_optimisation(pcm, bm, observable)
            IN:  pcm (n,nz), bm (D,nz), observable (d,d) Hermitian
            OUT: [variance (scalar), rho* (d,d), coefs* (n,)]
            Finds the state rho* that maximises the variance of the optimal
            estimator. This is the main result of the whole pipeline.
        - fix_coef_var_optimisation(pcm, bm, coefs)
            IN:  pcm (n,nz), bm (D,nz), coefs (n,) fixed coefficient vector
            OUT: [variance (scalar), rho* (d,d)]
            For a fixed coefficient vector (e.g. the canonical one), finds the
            worst-case state and its variance. Useful for benchmarking.

  EXPERIMENT SCRIPT — copy and adapt this for a new problem:

  plane_proj_XZ.py
      Sweeps over a family of observables (single-qubit states tensored N times,
      parametrised by theta in the XZ plane) and computes both the optimal and
      canonical variance for each. Loops over qubit counts N from 1 to N_max.

      Key inputs at the top of the file:
        single_povm    — the single-qubit POVM; here pf.pauli_povm_single('X','Z'),
                         a (4,2,2) array of XZ-plane projectors
        N_min, N_max   — range of qubit counts
        density        — number of theta values sampled in [0, pi/4]
        USE_PRODUCT_PURE — 0 for full state optimisation, 1 for product pure

      Outputs (saved to plane_proj_XZ_jax/):
        {dir}_{N}_opt.npy — (density,) array of optimal variances over theta
        {dir}_{N}_can.npy — scalar canonical variance (reference)
        time_log.npy      — (N_max,) runtimes per N
        plane_proj.pdf    — three-panel summary figure

      This is the file to copy when setting up a new problem.


How to run the existing experiment
-----------------------------------
  python plane_proj_XZ.py

Results are saved to the plane_proj_XZ_jax/ directory. Runtime is printed per
qubit count N; expect a few seconds per N on CPU for density=2, N_max=5.


How to adapt for a new problem
--------------------------------
Copy plane_proj_XZ.py and change three things:

1. Define your POVM.
   Write a function analogous to pauli_povm_single() that returns your POVM as an
   (n, d, d) array of effects summing to the identity. Pass it to
   pf.povm_coef_matrix() to get pcm and bm — these two are the only things the
   optimisation needs to know about your POVM.

2. Define your observable family.
   Replace sf.qubit(theta, phi) / pf.tensor_same(singleobs, N) with your observable
   as a (d, d) Hermitian matrix. Pass it to ocf.variance_optimisation(pcm, bm, obs).

3. Adjust N_min, N_max, density, and the output directory name.

Everything else (the optimisation, state parameterisation, basis construction)
runs unchanged.


Key variables / naming conventions
------------------------------------
  d       dimension of the single-system Hilbert space (d=2 for a qubit)
  D = d^2 dimension of the Hilbert-Schmidt space (space of d×d matrices, vectorised)
  n       number of POVM effects
  nz      dimension of the subspace spanned by the POVM effects (nz <= D)
  pcm     POVM coefficient matrix, shape (n, nz)
  bm      basis matrix for the POVM subspace, shape (D, nz), orthonormal columns
  flat_X  a matrix X vectorised into a 1D array of length D or nz (depending on context)
  coefs   the estimator coefficient vector c, same length as the number of POVM outcomes


Notes / known issues
---------------------
- The Adam optimiser may not converge to the global optimum for large N (many qubits)
  because the landscape becomes increasingly non-convex. The default ADAM_STEPS=200
  and ADAM_LR=2e-2 (set at the top of opt_variance_coef.py) were tuned for N<=5 qubits;
  for larger N, it is worth increasing the number of steps and checking convergence by
  running with different random initialisations.
- The uniform initialisation x0 = ones(D)/D corresponds to a state close to the maximally
  mixed state, which is a reasonable starting point but may miss local optima near pure states.
  If results look suspicious, try a few different x0 and take the maximum.
- JAX requires jax_enable_x64=True for complex128 arithmetic; this is set automatically in
  the experiment scripts but needs to be done before any JAX operations if running interactively.

================================================================================
