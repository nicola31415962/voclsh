# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Plane projectors optimal variance
# Let us consider one of the simplest cases: a (tensor) projector on a reduced subspace

# +
import pickle
import numpy as np

import scipy as sc
from scipy import io
from scipy import optimize
from joblib import Parallel, delayed

import time
import os

# Graphics
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
#rcParams['pcolor.shading']= 'auto'
import matplotlib.patches as patches
from matplotlib.colors import TwoSlopeNorm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.gridspec as gridspec

cm = 1.0/2.54  # centimeters in inches
px = 1.0/plt.rcParams['figure.dpi']  # pixel in inches

goldenratio=1.618
imagewidth = 7 # 
# -

# homebrewed modules for the actual variance optimisation and handling of different functions
import opt_variance_coef as ocf  # variance optimisation related functions
import states_functions as sf    # state and observables related functions
import povm_functions as pf      # POVM generation and handling related functions

# +
# global variables
single_povm = pf.pauli_povm_single('X','Z') # single qubit plane POVM

density = 2  # number of different projectors considered

N_min = 1     # minimal dimension considered
N_max = 3    # maximal dimension considered

# saving directory (created if non-existing)
directory= 'plane_proj_XZ'
os.system(f'mkdir {directory}')
# -

# # Cycle over different dimensions

# +
all_vec = []
all_can = []

min_vec = []
avg_vec = []

times = []

# --- helper function for parallel theta evaluation ---
def compute_var_for_theta(j, density, N, pcm, bm):
    theta = j*np.pi/(4*density)
    phi = 0
    singleobs = sf.qubit(theta, phi)
    obs = pf.tensor_same(singleobs, N)
    var, rho, oc = ocf.variance_optimisation(pcm, bm, obs)  # no 'method'
    return var

# --- main loop ---
for N in range(1, N_max+1):
    time_start = time.time()
    filename = directory + f'_{N}'

    # tensor POVM definition
    povm = pf.tensor_same(single_povm, N)
    [pcm, bm] = pf.povm_coef_matrix(povm)
    can_em = sc.linalg.pinv(pcm)  # canonical inverse

    # canonical inversion
    single_obs = sf.qubit(0, 0)
    obs = pf.tensor_same(single_obs, N)
    cc = can_em.T @ (sf.flatten_in_basis(obs, bm))
    [var_can, rho_can] = ocf.fix_coef_var_optimisation(pcm, bm, cc)
    np.save(f'{directory}/{filename}_can', var_can)

    # optimal upper bound on variance (parallelized over theta)
    var_vec = Parallel(n_jobs=-1)(
        delayed(compute_var_for_theta)(j, density, N, pcm, bm)
        for j in range(density)
    )
    np.save(f'{directory}/{filename}_opt', var_vec)

    all_vec.append(var_vec)
    all_can.append(var_can)
    min_vec.append(min(var_vec))
    avg_vec.append(np.average(np.array(var_vec)))

    time_N = time.time()
    print(N, time_N-time_start)
    times.append(time_N-time_start)

np.save(f'{directory}/time_log', times)
# -

# # Plots
# Let's make now something pretty to concisely see the results

# +
color_lite = [  'xkcd:azure', 'xkcd:kelly green','xkcd:bright orange','xkcd:red', 'xkcd:lavender', 'xkcd:golden yellow','xkcd:maroon','xkcd:light grey','xkcd:robin egg blue', 'xkcd:medium brown'] # most similar to std plt pallet
color_lite_dark = ['xkcd:royal blue',  'xkcd:forest green','xkcd:burnt orange', 'xkcd:scarlet', 'xkcd:violet', 'xkcd:mustard','xkcd:steel grey','xkcd:bright sky blue','xkcd:earth'] # most similar to std plt pallet

thetas = np.pi*np.arange(density)/(2*density)

nrow = 1
ncol = 3
fig,ax = plt.subplots(nrow, ncol, figsize=[ncol*imagewidth,nrow*imagewidth/goldenratio])

ax[0].set_title('Optimal variance')
ax[1].set_title('ratio with canonical case')
ax[0].set_ylabel(r'Var')
ax[1].set_ylabel(r'Var$_{opt}/$ Var$_{can}/$')
ax[0].set_xlabel(r'$\theta$')
ax[1].set_ylabel(r'$\theta$')


for N in range(N_max):
    ax[0].plot(thetas, all_vec[N], c= color_lite[N],label=f'{N+1} qubits')
    ax[0].hlines(all_can[N],0,np.pi/2,color = color_lite_dark[N], linestyle='--')
    
    ax[1].plot(thetas, all_vec[N]/all_can[N], c= color_lite[N], label=f'{N+1} qubits')
    
ax[2].set_title('scaling with number of qubits')
ax[2].plot(np.arange(1,N_max+1), min_vec, marker='o', label='best optimal scaling')
ax[2].plot(np.arange(1,N_max+1), avg_vec, marker='x',linestyle='--',label='average optimal scaling')
ax[2].plot(np.arange(1,N_max+1), all_can, marker='^',linestyle=':',label='canonical scaling')
    
ax[2].set_ylabel('min Var')
ax[2].set_xlabel('# qubits')

    
plt.savefig(f'{directory}/plane_proj.pdf', transparent=True, bbox_inches='tight')
# -
