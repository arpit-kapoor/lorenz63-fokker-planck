import argparse
import jax
import jax.numpy as jnp
import matplotlib as mlp
import matplotlib.pyplot as plt
import numpy as np
import os
import time

from datetime import datetime as dt
from jax import random
from joblib import Parallel, delayed
from matplotlib.animation import FuncAnimation
from tqdm import tqdm


start = time.time()
# --------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument('--M', type=int, default=40,
                    help='gap between observations')
# The num_particles default was 100000.
parser.add_argument('--num_particles', type=int, default=25,
                    help='no of particles')
# The batchsize default was 20000.
parser.add_argument('--batchsize', type=int, default=5,
                    help='no of particles in a batch')
parser.add_argument('--lhood-axis', type=int, default=2,
                    help='likelihood reduction 0 - z1, 1 - z2, 2 - mean')
parser.add_argument('--sy', type=float, default=0.5)
parser.add_argument('--savecode', type=int, default=2,
                    help='Save code: (0) array, (1) animation (2) both')
parser.add_argument('--comppost', type=int, default=1,
                    help='Compute posterior: (0) False, (1) True')

args = parser.parse_args()


# --------------------------------------------------------
# CONSTANTS
# --------------------------------------------------------

cmap = plt.cm.viridis

# Lorentz parameters.
S = 10
R = 28
B = 8/3

# Simulation parameters.
c = 0.1                             # Parameter in obs operator.
h = 1e-3                            # Time step for Euler discretisation.
M = args.M                          # Interobservation gap, in multiples of h.
# Tf = 20                           # Final time.
NT = int(20/h)                      # No. of discrete time steps.
sx = 0.05                           # Scale of signal noise.
sy = 0.5                            # Std of observation noise.
s2o = 1                             # Initial variance.
# NR = 1                            # No. of repetitions of the simulation.
# sw = [1 1]                        # Algorithms to run:
#                                       sw(1) is BF, sw(2) is CKF.
npts = 40                           # No. of points in each dimension in grid.
num_particles = args.num_particles  # No. of particles, must increase to 10^6.

batchsize = args.batchsize
d = 3                               # Dimension of state vector.
p = 2                               # Dimension of observation vector.
comppost = args.comppost            # = 1 means to compute posterior,
#                                     = 0 means only compute Fokker-Planck.
XR0 = jnp.array([[-5.91652],
                 [-5.52332],
                 [24.5723]])        # Reference initial condition
#                                     (also prior mean).
XR02 = jnp.array([[5.91652],
                  [5.52332],
                  [24.5723]])       # Other mode of initial distribution.

ZR0 = jnp.zeros((p, 1))             # Initial observation.


# Grid extents
extents = jnp.array([[-20, 20],
                     [-30, 20],
                     [0, 50]])

# Discrete time-steps where the observations are available. M-steps apart.
idy = jnp.arange(0, NT, M)


key = random.PRNGKey(1234)  # Random seed is explicit in JAX.

# --------------------------------------------------------
# SETUP LOGGING
# --------------------------------------------------------


def log_message(msg):
    print(f"{dt.now()}: {msg}")


# Output Directory.
#
# Colons aren't compatible with Windows file names.
# rundir = f'runs/run_{num_particles}_{dt.now().strftime("%Y-%m-%dT%H:%M:%S")}'
rundir = f'runs/run_{num_particles}_{dt.now().strftime("%Y-%m-%d_T%H-%M-%S")}'
# File to store runtime arguments.
args_file = os.path.join(rundir, 'args.txt')
# File to store the mesh the Fokker-Planck equation is solved over.
mesh_arr_file = os.path.join(rundir, 'mesh.npy')
# Files for PDE evolution animations.
prior_file = os.path.join(rundir, 'prior.gif')
lhood_file = os.path.join(rundir, 'lhood.gif')
post_file = os.path.join(rundir, 'post.gif')
# Files for PDE evolution arrays.
prior_arr_file = os.path.join(rundir, 'prior.npy')
lhood_arr_file = os.path.join(rundir, 'lhood.npy')
post_arr_file = os.path.join(rundir, 'post.npy')

log_message(f"Setting run directory to: {rundir}")

if os.path.exists(rundir):
    log_message("Run directory exists, continuing...")
else:
    log_message("Run directory not found, creating the directory now...")
    os.makedirs(rundir)
    log_message("Run directory created.")

with open(args_file, 'w') as fil:
    fil.write(str(args))


# --------------------------------------------------------
# VECTORISE PARTICLE MOBILITY COMPUTATION
# --------------------------------------------------------

# Matrices.
A1 = jnp.array([[-S, S, 0], [R, -1, 0], [0, 0, -B]])
A2 = jnp.zeros((3, 3)).at[0, 2].set(-1)
A3 = jnp.zeros((3, 3)).at[0, 1].set(1)

# Vectors.
a1 = jnp.zeros((3, 1)).at[1, 0].set(1)
a2 = jnp.zeros((3, 1)).at[2, 0].set(1)


# --------------------------------------------------------
# CORE FUNCTIONS
# --------------------------------------------------------

@jax.jit
def nu(X):
    """Function to compute mobility at each time
    given the current particle position"""
    mat1 = A1 @ X
    mat2 = jnp.sum((X.T @ A2) * X.T, axis=1) * a1
    mat3 = jnp.sum((X.T @ A3) * X.T, axis=1) * a2
    return mat1 + mat2 + mat3


@jax.jit
def hobs_l63(X, c):
    """Transform particle position (3-dim) to obervation space (2-dim)"""
    Z0 = c*X[0, :]*X[1, :]
    Z1 = c*X[2, :]
    Z = jnp.stack([Z0, Z1], axis=0)
    return Z


@jax.jit
def time_update(carry, t):
    """Evolve states (XR) and observations (ZR) in time"""
    carry['ZT'] = carry['ZT'] + h*hobs_l63(carry['XT'], c) \
        + sy*np.sqrt(h)*np.random.randn(p, 1)
    carry['XT'] = carry['XT'] + h*nu(carry['XT']) \
        + sx*np.sqrt(h)*np.random.randn(d, 1)

    return carry, carry


@jax.jit
def evolve_particles(i, X):
    """Evolve particles in orginal 3-dim space with time"""
    num_particles_batch = X.shape[-1]
    X = X + h*nu(X) + sx*np.sqrt(h)*np.random.randn(d, num_particles_batch)
    return X


@jax.jit
def computepdf(X):
    """Compute prior PDF over the entire mesh grid
    for given particle position"""
    return jax.scipy.stats.multivariate_normal.pdf(
                mesh, mean=X, cov=0.1*jnp.identity(d)
    )


@jax.jit
def fokker_planck(carry, idx):
    """Fokker-Planck function to evolve particles and compute
    the PDF for the mesh at evolved states
    """
    carry['XT'] = jax.lax.fori_loop(0, M, evolve_particles, carry['XT'])
    log_message("Particles evolved.")
    priorpdfi = jnp.apply_along_axis(computepdf,
                                     axis=0,
                                     arr=carry['XT']).sum(axis=-1)
    log_message("PDF computed.")
    return carry, priorpdfi


# --------------------------------------------------------
# PLOTTING FUNCTIONS
# --------------------------------------------------------

def scale(x):
    x_max = x.max()
    x_min = x.min()
    return (x - x_min) / (x_max - x_min)


def plot_pdf(x, y, z, p0, pn, name='plot.gif'):

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'),
                           constrained_layout=True)

    norm = mlp.colors.Normalize(vmin=p0.min(), vmax=p0.max())
    scat = ax.scatter(x, y, z, s=scale(p0)*2000, c=cmap(norm(p0)),
                      alpha=0.5)

    # ax.set_zlim([0,np.max(Pt)/3])
    ax.autoscale(False)

    def update(i):
        norm = mlp.colors.Normalize(vmin=pn[i].min(),
                                    vmax=pn[i].max())
        scat._offsets3d = (x, y, z)
        scat.set_color(cmap(norm(pn[i].ravel())))
        scat.set_sizes(scale(pn[i].flatten()) * 1000)
        return [scat]

    anim = FuncAnimation(fig, update, frames=tqdm(range(len(pn))), interval=1)
    ax.set(xlabel='x1', ylabel='x2', zlabel='x3')

    # Saving to m4 using ffmpeg writer.
    anim.save(name, fps=60)

    plt.close()


def plot_landscape(x, y, z, p0, pn, xarr, name='plot.gif'):

    # Reduce the number of frames to 499.
    skip = pn.shape[0] // 499
    pn = pn[::skip]

    def format_str(x):
        lis = x.tolist()
        return f"{lis[0]:.2f}, {lis[1]:.2f}, {lis[2]:.2f}"

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'),
                           constrained_layout=True)

    norm = mlp.colors.Normalize(vmin=p0.min(), vmax=p0.max())
    scat = ax.scatter(x, y, z, s=scale(p0)*2000, c=cmap(norm(p0)),
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
        scatx._offsets3d = (xarr[0, i+1][None], xarr[1, i+1][None],
                            xarr[2, i+1][None])
        scatx.set_sizes([20])
        ax.set_title(format_str(xarr[:, i+1]))
        return [scat]

    anim = FuncAnimation(fig, update, frames=tqdm(range(len(pn))), interval=0)
    ax.set(xlabel='x1', ylabel='x2', zlabel='x3')

    # Saving to m4 using ffmpeg writer.
    anim.save(name, fps=60)

    plt.close()


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

# Define timesteps and carry vectors.
log_message("Reference solution...")

# NT is the total number of timesteps.
# Array of discrete time-steps.
t = jnp.arange(NT)
carry = {'XT': XR0, 'ZT': ZR0}

# Simulate the Lorenz63 system to generate a reference solution.
carry_out, out = jax.lax.scan(time_update, carry, t)

# Process outputs.
XR = out['XT'].T.reshape((d, -1))
ZR = out['ZT'].T.reshape((p, -1))

log_message("Done.")

# Generate mesh-grid to evaluate the probabilities over.
points = [jnp.linspace(*extents[i], npts) for i in range(d)]
mesh = jnp.stack(jnp.meshgrid(*points), axis=-1)


# --------------------------------------------------------
# COMPUTE PRIOR DENSITIES
# --------------------------------------------------------

log_message("Generate prior pdf...")

# Initialise particles by sampling from the initial distributions.
# Half of the particles are sampled from Multivariate Gaussian with mean XR0.
# The other half are sampled from Multivariate Gaussian with mean XR02.
cov = jnp.identity(d) * s2o

# Particles set-1 (XR0).
XP1 = random.multivariate_normal(mean=XR0.flatten(),
                                 cov=cov,
                                 shape=(int(num_particles*0.5),),
                                 key=key).T

# Particles set-2 (XR02).
XP2 = random.multivariate_normal(mean=XR02.flatten(),
                                 cov=cov,
                                 shape=(int(num_particles*0.5),),
                                 key=key).T

# Concatenate the two sets of particles.
XP = jnp.concatenate([XP1, XP2], axis=1)


def parallelfunc(XP_batch, input_evolve):
    """Function to compute the prior densities for a batch of particles
    """
    # Compute the densities for initial position of the particles.
    priorpdf0 = jnp.apply_along_axis(computepdf, axis=0,
                                     arr=XP_batch).sum(axis=-1)

    # Evolve and compute densities.
    #
    # Parameter 2: initial position of the particles.
    # Parameter 3: evolve the particles for idy-1 steps.
    carry, priorpdfn = jax.lax.scan(
        fokker_planck,
        {"XT": XP_batch},
        input_evolve)

    # Concatenate initial pdf with pdf at remaining time-steps.
    return jnp.concatenate(
        [jnp.expand_dims(priorpdf0, axis=0), priorpdfn], axis=0)


# This is where I need help...
# I use joblib delayed to parallelize `parallelfunc`.
# The particles (XP) are divided into batches, and the prior densities for each
# batch is computed in parallel. Finally, we add the densities computed for
# individual batches to get the final densities.

t1 = time.time()

# Spawn several jobs to run the prior density calculation in parallel.
param_evolve = jnp.arange(len(idy) - 1)
results = Parallel(
    n_jobs=min(os.cpu_count(), round(num_particles / batchsize)))(
              delayed(parallelfunc)(
                XP[:, i: i+batchsize], param_evolve) for i in range(
                  0, num_particles, batchsize))
# Add the results to get final pdf over the mesh-grid.
# Note, it's assumed that "sum" is acting as a synchronisation barrier, but
#    it'd be prudent to confirm that.
priorpdfn = sum(results)
priorpdf0 = priorpdfn[0]
priorpdfn = priorpdfn[1:]
t2 = time.time()
log_message(f"Time taken for prior density: {t2 - t1:.1f} s")


# --------------------------------------------------------
# COMPUTE POSTERIOR
# --------------------------------------------------------

# Multiply by likelihood to get posterior.
if comppost:
    log_message("Generate posterior...")
    ZL = ZR[:, idy]
    dpoints = mesh.transpose((3, 0, 1, 2)).reshape((d, -1))

    hdpts = hobs_l63(dpoints, c)   # Convert to observation space.
    y = jnp.expand_dims(jnp.diff(ZL, axis=1).T/(M*h), axis=-1)

    if args.lhood_axis == 2:
        correc = jnp.exp(
            -0.5 * (M*h) * jnp.square(
                jnp.repeat(y, hdpts.shape[-1], axis=-1) - hdpts
            ).sum(axis=1)
        )
    else:
        correc = jnp.exp(
            -0.5 * (M*h) * jnp.square(
                jnp.repeat(y, hdpts.shape[-1], axis=-1) - hdpts
            )[:, args.lhood_axis]
        )
    correc = correc.reshape((-1, npts, npts, npts))
    proppdfn = priorpdfn * correc


# --------------------------------------------------------
# SAVE ARRAY: Write arrays to file
# --------------------------------------------------------

if args.savecode in [0, 2]:

    # Mesh grid.
    with open(mesh_arr_file, 'wb') as f:
        jnp.save(f, mesh)

    # Prior PDF.
    with open(prior_arr_file, 'wb') as f:
        jnp.save(f, priorpdfn)

    if comppost:

        # Likelihood PDF.
        with open(lhood_arr_file, 'wb') as f:
            jnp.save(f, correc)

        # Posterior PDF.
        with open(post_arr_file, 'wb') as f:
            jnp.save(f, proppdfn)


# --------------------------------------------------------
# GENERATE PLOTS
# --------------------------------------------------------

if args.savecode in [1, 2]:

    x = mesh[:, :, :, 0].ravel()   # xmesh.
    y = mesh[:, :, :, 1].ravel()   # ymesh.
    z = mesh[:, :, :, 2].ravel()   # zmesh.

    # Plot prior pdf and save to file.
    log_message("Plotting prior density map.")
    p0 = priorpdf0.ravel()
    plot_pdf(x, y, z, p0, priorpdfn, prior_file)

    # Plot posterior pdf and save to file.
    if comppost:
        log_message("Plotting likelihood landscape.")
        correc0 = correc[0].ravel()
        plot_landscape(x, y, z, correc0, correc, XR[:, idy], lhood_file)

        log_message("Plotting posterior density map.")
        plot_pdf(x, y, z, p0, proppdfn, post_file)


# --------------------------------------------------------
# FINALISE
# --------------------------------------------------------
log_message("Done.")
end = time.time()
log_message(f"Execution time: {end - start:.1f} s")
