import jax.lax as lax
from einops import repeat
import jax.numpy as np
from matplotlib.figure import Figure

from utils import splitkey
from model import prior, gen
from deepset import masksum
from utils import dmap, dmap1

def plot(keyrest, nbatches, phi, rho, dparams, nmax, prefix="", label="", ntrain=-1, groundtruth=None):

  key , keyrest = splitkey(keyrest)
  labels = prior(key, nbatches)
  key , keyrest = splitkey(keyrest)
  obs , _ = gen(keyrest, labels, nmax)
  embedding = dmap1(lambda p: phi.apply(p["phi"], lax.stop_gradient(obs)), dparams)

  arr = { k : [] for k in dparams }
  for n in range(1, nmax+1):

    ns = repeat(np.array([n]), "w -> h w", h=nbatches, w=1)
    summed = dmap1(lambda e: masksum(e, ns), embedding)

    predicted = dmap(lambda k, s: rho.apply(dparams[k]["rho"], s), summed)
    predicted = dmap1(lambda p : p.at[:,1:].set(np.exp(p[:,1:])), predicted)

    for k in dparams:
      meanbiasmu = np.mean(predicted[k][:,0] - labels[:,0])
      meanuncertmu = np.mean(predicted[k][:,1])

      arr[k].append \
        ( [ meanbiasmu
          , meanuncertmu
          ]
        )

  def savepdf(fig, name):
    return fig.savefig(f"{prefix}{name}{label}.pdf")

  fig = Figure((6, 6))
  plt = fig.add_subplot()
  for k in dparams:
    plt.plot(range(1, nmax+1), [ a[0] for a in arr[k] ], label=k)

  if (ntrain > 0):
    ymin, ymax = plt.get_ylim()
    plt.plot([ntrain]*2, [ymin, ymax], alpha=0.5, color="black", ls=":", label="max $\phi$ training", zorder=10)

  plt.set_xlabel(r"number of observations")
  plt.set_ylabel(r"mean bias of $\mu$")
  plt.legend()
  fig.tight_layout()
  savepdf(fig, "meanbiasmu")
  fig.clf()

  ns = np.mgrid[1:nmax+1:nmax*1j]
  fig = Figure((6, 6))
  plt = fig.add_subplot()

  for k in dparams:
    plt.plot(range(1, nmax+1), [ a[1] for a in arr[k] ], label=k)

  if groundtruth is not None:
    plt.plot(ns, groundtruth(ns), ls="--", color="red", label="ground truth")

  plt.set_xlabel(r"number of observations")
  plt.set_ylabel(r"mean posterior width")
  plt.loglog()

  if (ntrain > 0):
    ymin, ymax = plt.get_ylim()
    plt.plot([ntrain]*2, [ymin, ymax], alpha=0.5, color="black", ls=":", label="max $\phi$ training", zorder=10)

  plt.legend()
  fig.tight_layout()
  savepdf(fig, "meanuncertmu")
  fig.clf()
