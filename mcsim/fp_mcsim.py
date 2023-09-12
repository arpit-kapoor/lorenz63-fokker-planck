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
comppost = 1                                #= 1 means to compute posterior, = 0 means only compute fokker planck
XR0 = jnp.array([[-5.91652], 
                 [-5.52332], 
                 [24.5723]])               # reference initial condition (also prior mean)

XR02 = jnp.array([[5.91652], 
                  [5.52332], 
                  [24.5723]])               # other mode of initial distribution 

ZR0 = jnp.zeros((p, 1))


def log_message(msg):
    print(f"{dt.now()}: {msg}")


# Output Dir
rundir = f'runs/run_{NP}_{dt.now().strftime("%Y-%m-%dT%H:%M:%S")}'
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


# Grid extents
extents = jnp.array([[-20, 20],
                     [-30, 20],
                     [0, 50]])

idy = jnp.arange(0, NT, M)

# Matrices
A1 = jnp.array([[-S, S, 0], [R, -1, 0], [0, 0, -B]])
A2 = jnp.zeros((3, 3)).at[0, 2].set(-1)
A3 = jnp.zeros((3, 3)).at[0, 1].set(1)

# Vectors
a1 = jnp.zeros((3, 1)).at[1, 0].set(1)
a2 = jnp.zeros((3, 1)).at[2, 0].set(1)

key = random.PRNGKey(1234)  # Random seed is explicit in JAX

# --------------------------------------------------------


@jax.jit
def nu(X):
    mat1  = A1 @ X
    mat2 = jnp.sum((X.T @ A2) * X.T, axis=1) * a1
    mat3 = jnp.sum((X.T @ A3) * X.T, axis=1) * a2
    return  mat1 + mat2 + mat3

@jax.jit
def hobs_l63(X, c):
    Z0 = c*X[0, :]*X[1,:]
    Z1 = c*X[2, :]
    Z = jnp.stack([Z0, Z1], axis=0)
    return Z

@jax.jit
def time_update(carry, t):
    carry['ZT'] = carry['ZT'] + h*hobs_l63(carry['XT'], c) +\
                                sy*np.sqrt(h)*np.random.randn(p,1)
    carry['XT'] = carry['XT'] + h*nu(carry['XT']) +\
                                sx*np.sqrt(h)*np.random.randn(d, 1)

    return carry, carry


@jax.jit
def evolve_particles(i, X):
    NP_batch = X.shape[-1]
    X = X + h*nu(X) + sx*np.sqrt(h)*np.random.randn(d, NP_batch)
    return X

@jax.jit
def computepdf(X):
    bw = 0.1
    return jax.scipy.stats.multivariate_normal.pdf(
                mesh, mean=X, cov=bw*jnp.identity(d)
    )

@jax.jit
def fokker_planck(carry, idx):
    carry['XT'] = jax.lax.fori_loop(0, M, evolve_particles, carry['XT'])
    log_message("Particles evolved")
    priorpdfi = jnp.apply_along_axis(computepdf, 
                                     axis=0, 
                                     arr=carry['XT']).sum(axis=-1)
    log_message("PDF computed")
    return carry, priorpdfi

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
# --------------------------------------------------------


# Define timesteps and carry vectors
log_message("Reference solution..")

t = jnp.arange(NT)
carry = {
    'XT': XR0,
    'ZT': ZR0
}

# Lorenz 63 simulation
carry_out, out = jax.lax.scan(time_update, carry, t)

# Process outputs
XR = out['XT'].T.reshape((d, -1))
ZR = out['ZT'].T.reshape((p, -1))

log_message("Done!")

# Mesh grid
points = [jnp.linspace(*extents[i], npts) for i in range(d)]
mesh = jnp.stack(jnp.meshgrid(*points), axis=-1)

log_message("Generate prior pdf ..")

#Initial pdf
cov = jnp.identity(d) * s2o
XP1 = random.multivariate_normal(mean=XR0.flatten(), 
                                cov=cov, 
                                shape=(int(NP/2),), 
                                key=key).T
XP2 = random.multivariate_normal(mean=XR02.flatten(), 
                                cov=cov, 
                                shape=(int(NP/2),), 
                                key=key).T

XP = jnp.concatenate([XP1, XP2], axis=1)

def parallelfunc(XP_batch):
    priorpdf0 = jnp.apply_along_axis(computepdf, 
                        axis=0, 
                        arr=XP_batch).sum(axis=-1)
    idx = jnp.arange(len(idy)-1)
    carry_in = {
        'XT': XP_batch
    }
    carry, priorpdfn = jax.lax.scan(fokker_planck, carry_in, idx)
    return jnp.concatenate([jnp.expand_dims(priorpdf0, axis=0), priorpdfn], axis=0)



# t1 = time.time()
# results = Parallel(n_jobs=8, backend='loky')(
#                 delayed(parallelfunc)(XP[:, i: i+batchsize]) for i in range(0, NP, batchsize))
# priorpdfn = sum(results)
# priorpdf0 = priorpdfn[0]
# priorpdfn = priorpdfn[1:]
# t2 = time.time()
# log_message(f"Time taken for prior density: {t2 - t1:.2}s")



# %step 3. multiply by likelihood to get posterior: 
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
    # proppdf = priorpdfn * correc

log_message("Done!")

xmesh = mesh[:, :, :, 0]
ymesh = mesh[:, :, :, 1]
zmesh = mesh[:, :, :, 2]

x = xmesh.ravel()
y = ymesh.ravel()
z = zmesh.ravel()


log_message("Plotting landscape")
correc0 = correc[0].ravel()
landscape_file = os.path.join(rundir,'landscape.gif')
plot_landscape(x, y, z, correc0, correc, XR[:, idy], landscape_file)

# # Plot prior pdf and save to file
# log_message("Plotting prior density map")
# p0 = priorpdf0.ravel()
# plot_pdf(x, y, z, p0, priorpdfn, prior_file)

# # Plot posterior pdf and save to file
# if comppost:
#     log_message("Plotting posterior density map")
#     plot_pdf(x, y, z, p0, proppdf, post_file)



log_message(f"Job completed!")

