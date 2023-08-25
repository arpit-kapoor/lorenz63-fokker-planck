# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import datetime as dt
import matplotlib.colors
from matplotlib.animation import FFMpegWriter, FuncAnimation
from fplanck import fokker_planck, boundary, gaussian_pdf
from mpl_toolkits.mplot3d import Axes3D

cmap = plt.cm.binary

def message(msg):
    print(f"{dt.datetime.now()} =====> {msg}")

# %%
dims = 3
s = 10
r = 28
b = 8/3
sigma = np.ones(dims) * 4/5

# %%
T = 0.5 * np.square(sigma)/sc.k
drag = np.ones(dims)

# %%
def force(x1, x2, x3):
    nu1 = -s * (x1 - x2)
    nu2 = r * x1 - x2 - x1 * x3
    nu3 = x1 * x2 - b * x3
    return np.stack([nu1, nu2, nu3])

# %%

message("Instantiating Fokker Planck")
extent = np.array([40, 40, 40])
resolution = np.ones(dims) * 1
sim = fokker_planck(temperature=T, drag=drag, extent=extent,
            resolution=resolution, boundary=boundary.reflecting, force=force)

message("Done, now propagating interval")


# %%
### time-evolved solution
pdf = gaussian_pdf(center=(-5.91652, -5.52332, 24.5723), width=5)
p0 = pdf(*sim.grid)

Nsteps = 100
time, Pt = sim.propagate_interval(pdf, 1e-3, Nsteps=Nsteps)


message("Done, now generating plot")


# %%
fig, ax = plt.subplots(subplot_kw=dict(projection='3d'), constrained_layout=True)

x = sim.grid[0].ravel()
y = sim.grid[1].ravel()
z = sim.grid[2].ravel()
p0 = p0.ravel()

norm = matplotlib.colors.Normalize(vmin=p0.min(), vmax=p0.max())
scat = ax.scatter(x, y, z, color=cmap(norm(p0)), alpha=0.1)

# ax.set_zlim([0,np.max(Pt)/3])
ax.autoscale(False)

def update(i):
    global scat
    print(i)
    scat.remove()
    scat = ax.scatter(x, y, z, color=cmap(norm(Pt[i].ravel())), alpha=0.1)
    return [scat]

anim = FuncAnimation(fig, update, frames=range(Nsteps), interval=10)
ax.set(xlabel='x1', ylabel='x2', zlabel='x3')

# # saving to m4 using ffmpeg writer
writervideo = FFMpegWriter(fps=60)
anim.save('fokkerplanckgauss5.mp4', writer=writervideo)

plt.close()


message("Done!!")
