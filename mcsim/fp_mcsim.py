import argparse
import os
import time
from joblib import Parallel, delayed
import jax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mlp
import jax.numpy as jnp

from datetime import datetime as dt
from matplotlib.animation import FuncAnimation
from jax import random
from tqdm import tqdm



# --------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--M', type=int, default=40, help='gap between observations')
parser.add_argument('--NP', type=int, default=100000, help='no of particles')
parser.add_argument('--batchsize', type=int, default=20000, help='no of particles in a batch')
parser.add_argument('--lhood-axis', type=int, default=2, help='likelihood reduction 0 - z1, 1 - z2, 2 - mean')
parser.add_argument('--sy', type=float, default=0.5)

args = parser.parse_args()


# --------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------

cmap = plt.cm.viridis

## Lorenz parameters
S = 10
R = 28
B = 8/3

## Simulation parameters
h = 1e-3                                    # time step for euler discretisation 
c = 1/10                                    # parameter in obs operator  
M = args.M                                      # gap between observations (in multiples of h)
Tf = 20                                     # final time
NT = int(Tf/h);  #1+floor(Tf/h)             # no. of discrete time steps
sx = 0.05                                   # scale of signal noise
sy = 0.5                                    # std of observation noise
s2o = 1                                     # initial variance
#NR = 1;                                    # no. of repetitions of the simulation
#sw = [1 1];                                # algorithms to run: sw(1) is BF, sw(2) is CKF
npts = 40                                   # no. of points in each dimension in grid
NP = args.NP                                # no. of particles, needs to be increased to 10^6 

batchsize = args.batchsize
d = 3;                                      # dimension of state vector 
p = 2;                                      # dimension of observation vector 
comppost = 0                                #= 1 means to compute posterior, = 0 means only compute fokker planck
XR0 = jnp.array([[-5.91652], 
                 [-5.52332], 
                 [24.5723]])               # reference initial condition (also prior mean)

XR02 = jnp.array([[5.91652], 
                  [5.52332], 
                  [24.5723]])               # other mode of initial distribution 

ZR0 = jnp.zeros((p, 1))                     # Initial obervation


# Grid extents
extents = jnp.array([[-20, 20],
                     [-30, 20],
                     [0, 50]])

idy = jnp.arange(0, NT, M)                  # Discrete time-steps where 
                                            # the observations are available
                                            # M-steps apart


key = random.PRNGKey(1234)  # Random seed is explicit in JAX

# --------------------------------------------------------
# SETUP LOGGING
# --------------------------------------------------------

def log_message(msg):
    print(f"{dt.now()}: {msg}")

# Output Dir
rundir = f'runs/run_{NP}_{dt.now().strftime("%Y-%m-%dT%H:%M:%S")}'
lhood_file = os.path.join(rundir,'lhood.gif')
prior_file = os.path.join(rundir,'prior.gif')
post_file = os.path.join(rundir,'post.gif')
args_file = os.path.join(rundir, 'args.txt')

log_message(f"Setting run directory to: {rundir}")

if os.path.exists(rundir):
    log_message("Run directory exists, continuing..")
else:
    log_message("Run directory not found, creating the directory now..")
    os.makedirs(rundir)
    log_message("Run directory created!")

with open(args_file, 'w') as fil:
    fil.write(str(args))




# --------------------------------------------------------
# VECTORIZE PARTICLE MOBILITY COMPUTATION
# --------------------------------------------------------

# Matrices
A1 = jnp.array([[-S, S, 0], [R, -1, 0], [0, 0, -B]])
A2 = jnp.zeros((3, 3)).at[0, 2].set(-1)
A3 = jnp.zeros((3, 3)).at[0, 1].set(1)

# Vectors
a1 = jnp.zeros((3, 1)).at[1, 0].set(1)
a2 = jnp.zeros((3, 1)).at[2, 0].set(1)


# --------------------------------------------------------
# CORE FUNCTIONS
# --------------------------------------------------------

@jax.jit
def nu(X):
    """Function to compute mobility at each time 
    given the current particle position"""
    mat1  = A1 @ X
    mat2 = jnp.sum((X.T @ A2) * X.T, axis=1) * a1
    mat3 = jnp.sum((X.T @ A3) * X.T, axis=1) * a2
    return  mat1 + mat2 + mat3


@jax.jit
def hobs_l63(X, c):
    """Transform particle position (3-dim) to obervation space (2-dim)"""
    Z0 = c*X[0, :]*X[1,:]
    Z1 = c*X[2, :]
    Z = jnp.stack([Z0, Z1], axis=0)
    return Z

@jax.jit
def time_update(carry, t):
    """Evolve states (XR) and observations (ZR) in time"""
    carry['ZT'] = carry['ZT'] + h*hobs_l63(carry['XT'], c) +\
                                sy*np.sqrt(h)*np.random.randn(p,1)
    carry['XT'] = carry['XT'] + h*nu(carry['XT']) +\
                                sx*np.sqrt(h)*np.random.randn(d, 1)

    return carry, carry


@jax.jit
def evolve_particles(i, X):
    """Evolve particles in orginal 3-dim space with time"""
    NP_batch = X.shape[-1]
    X = X + h*nu(X) + sx*np.sqrt(h)*np.random.randn(d, NP_batch)
    return X

@jax.jit
def computepdf(X):
    """Compute prior pdf over the entire mesh grid
    for given particle position"""
    bw = 0.1
    return jax.scipy.stats.multivariate_normal.pdf(
                mesh, mean=X, cov=bw*jnp.identity(d)
    )

@jax.jit
def fokker_planck(carry, idx):
    """Fokker planck function to evolve particles and compute 
    the pdf for the mesh at evolved states
    """
    carry['XT'] = jax.lax.fori_loop(0, M, evolve_particles, carry['XT'])
    log_message("Particles evolved")
    priorpdfi = jnp.apply_along_axis(computepdf, 
                                     axis=0, 
                                     arr=carry['XT']).sum(axis=-1)
    log_message("PDF computed")
    return carry, priorpdfi


# --------------------------------------------------------
# PLOTTING FUNCTIONS
# --------------------------------------------------------

def scale(x):
    x_max = x.max()
    x_min = x.min()
    return (x - x_min)/(x_max - x_min)

def plot_pdf(x, y, z, p0, pn, name='plot.gif'):
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), 
                           constrained_layout=True)

    norm = mlp.colors.Normalize(vmin=p0.min(), vmax=p0.max())
    scat = ax.scatter(x, y, z, s=scale(p0)*2000, 
                               color=cmap(norm(p0)), 
                               alpha=0.5)

    # ax.set_zlim([0,np.max(Pt)/3])
    ax.autoscale(False)

    def update(i):
        norm = mlp.colors.Normalize(vmin=pn[i].min(), 
                                    vmax=pn[i].max())
        scat._offsets3d = (x, y, z)
        scat.set_color(cmap(norm(pn[i].ravel())))
        scat.set_sizes(scale(pn[i].flatten())*1000)
        return [scat]

    anim = FuncAnimation(fig, update, frames=tqdm(range(len(pn))), interval=1)
    ax.set(xlabel='x1', ylabel='x2', zlabel='x3')

    # # saving to m4 using ffmpeg writer
    anim.save(name, fps=60) 

    plt.close()


def plot_landscape(x, y, z, p0, pn, xarr, name='plot.gif'):

    # Reduce the number of frames to 499
    skip = pn.shape[0]//499
    pn = pn[::skip]


    def format_str(x):
        lis = x.tolist()
        return f"{lis[0]:.2f}, {lis[1]:.2f}, {lis[2]:.2f}"
    
    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), 
                           constrained_layout=True)

    norm = mlp.colors.Normalize(vmin=p0.min(), vmax=p0.max())
    scat = ax.scatter(x, y, z, s=scale(p0)*2000, 
                               color=cmap(norm(p0)), 
                               alpha=0.5)
    scatx = ax.scatter(xarr[0, 0], xarr[1, 0], xarr[2, 0], color='red')
    ax.set_title(format_str(xarr[:, 0]))

    # ax.set_zlim([0,np.max(Pt)/3])
    ax.autoscale(False)

    def update(i):
        norm = mlp.colors.Normalize(vmin=pn[i].min(), 
                                    vmax=pn[i].max())
        scat._offsets3d = (x, y, z)
        scat.set_color(cmap(norm(pn[i].ravel())))
        scat.set_sizes(scale(pn[i].flatten())*1000)
        scatx._offsets3d = (xarr[0, i+1][None], xarr[1, i+1][None], xarr[2, i+1][None])
        scatx.set_sizes([20])
        ax.set_title(format_str(xarr[:, i+1]))
        return [scat]

    anim = FuncAnimation(fig, update, frames=tqdm(range(len(pn))), interval=0)
    ax.set(xlabel='x1', ylabel='x2', zlabel='x3')

    # # saving to m4 using ffmpeg writer
    anim.save(name, fps=60) 

    plt.close()


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

# Define timesteps and carry vectors
log_message("Reference solution..")

# NT is thhe total number of timesteps
# Array of discrete time-steps
t = jnp.arange(NT)
carry = {
    'XT': XR0,
    'ZT': ZR0
}

# Simulate the Lorenz63 system to generate reference solution
carry_out, out = jax.lax.scan(time_update, carry, t)

# Process outputs
XR = out['XT'].T.reshape((d, -1))
ZR = out['ZT'].T.reshape((p, -1))

log_message("Done!")

# Generate mesh-grid to evaluate the probabilities over
points = [jnp.linspace(*extents[i], npts) for i in range(d)]
mesh = jnp.stack(jnp.meshgrid(*points), axis=-1)

log_message("Generate prior pdf ..")


# --------------------------------------------------------
# COMPUTE PRIOR DENSITIES
# --------------------------------------------------------

# Initialize particles by sampling from the initial distributions
# Half of the particles are sampled from Multivariate Gaussian with mean XR0
# while the other half are sampled from Multivariate Gaussian with mean XR02
cov = jnp.identity(d) * s2o

# Particles set-1 (XR0)
XP1 = random.multivariate_normal(mean=XR0.flatten(), 
                                cov=cov, 
                                shape=(int(NP/2),), 
                                key=key).T

# Particles set-2 (XR02)
XP2 = random.multivariate_normal(mean=XR02.flatten(), 
                                cov=cov, 
                                shape=(int(NP/2),), 
                                key=key).T

# Concatenate the two sets of particles
XP = jnp.concatenate([XP1, XP2], axis=1)


def parallelfunc(XP_batch):
    """Function to compute the prior densities for a batch of particles
    """
    # Compute the densities for initial position of the particles
    priorpdf0 = jnp.apply_along_axis(computepdf, 
                        axis=0, 
                        arr=XP_batch).sum(axis=-1)
    
    # Evolve the particles for idy-1 steps
    idx = jnp.arange(len(idy)-1)

    # Initial position of the particles
    carry_in = {
        'XT': XP_batch
    }

    # Evolve and compute densities
    carry, priorpdfn = jax.lax.scan(fokker_planck, carry_in, idx)

    # Concatenate initial pdf with pdf at remaining time-steps
    return jnp.concatenate([jnp.expand_dims(priorpdf0, axis=0), priorpdfn], axis=0)


### This is where I need help...
### I joblib delayed to parallelize `parallelfunc`
### The particles (XP) are divided into batches and the prior densities
###### for each batch is computed in parallel. Finally, we add the densities
###### computed for individual batches to get the final densities

t1 = time.time()

# Spawn njobs to run the prior density calculation in parallel
results = Parallel(n_jobs=8)(
                delayed(parallelfunc)(XP[:, i: i+batchsize]) for i in range(0, NP, batchsize))
# Add the results to get final pdf over the mesh-grid
priorpdfn = sum(results)
priorpdf0 = priorpdfn[0]
priorpdfn = priorpdfn[1:]
t2 = time.time()
log_message(f"Time taken for prior density: {t2 - t1:.2}s")



# --------------------------------------------------------
# COMPUTE POSTERIOR
# --------------------------------------------------------

# Multiply by likelihood to get posterior: 
if comppost:
    log_message("Generate posterior...")
    ZL = ZR[:, idy]
    dpoints = mesh.transpose((3, 0, 1, 2)).reshape((d, -1))

    hdpts = hobs_l63(dpoints, c)  #convert to observation space 
    y = jnp.expand_dims(jnp.diff(ZL, axis=1).T/(M*h), axis=-1)

    if args.lhood_axis == 2:
        correc = jnp.exp(-0.5*(M*h)*jnp.square(jnp.repeat(y, hdpts.shape[-1], axis=-1) - hdpts).sum(axis=1))
    else:
        correc = jnp.exp(-0.5*(M*h)*jnp.square(jnp.repeat(y, hdpts.shape[-1], axis=-1) - hdpts)[:, args.lhood_axis])
    correc = correc.reshape((-1, npts, npts, npts))
    proppdfn = priorpdfn * correc

log_message("Done!")



# --------------------------------------------------------
# GENERATE PLOTS
# --------------------------------------------------------


xmesh = mesh[:, :, :, 0]
ymesh = mesh[:, :, :, 1]
zmesh = mesh[:, :, :, 2]

x = xmesh.ravel()
y = ymesh.ravel()
z = zmesh.ravel()


log_message("Plotting likelighood landscape")
correc0 = correc[0].ravel()
plot_landscape(x, y, z, correc0, correc, XR[:, idy], lhood_file)

# Plot prior pdf and save to file
log_message("Plotting prior density map")
p0 = priorpdf0.ravel()
plot_pdf(x, y, z, p0, priorpdfn, prior_file)

# Plot posterior pdf and save to file
if comppost:
    log_message("Plotting posterior density map")
    plot_pdf(x, y, z, p0, proppdfn, post_file)

