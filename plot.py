import jax.lax as lax
from einops import repeat
import jax.numpy as np
from matplotlib.figure import Figure
from utils import splitkey
from model import prior, gen, SIGMAMU, SIGMAX
from deepset import fwd, masksum

def plot(keyrest, batches, phi, rho, params, nmax, prefix="", label="", ntrain=-1, groundtruth=None):
  arr = []

  key , keyrest = splitkey(keyrest)
  labels = prior(key, batches)
  key , keyrest = splitkey(keyrest)
  obs , _ = gen(keyrest, labels, nmax)
  embedding = phi.apply(params["phi"], lax.stop_gradient(obs))

  for n in range(1, nmax+1):

    ns = repeat(np.array([n]), "w -> h w", h=batches, w=1)
    summed = masksum(embedding, ns)

    predicted = rho.apply(params["rho"], summed)
    predicted = predicted.at[:,1:].set(np.exp(predicted[:,1:]))

    meanbiasmu = np.mean(predicted[:,0] - labels[:,0])
    meanuncertmu = np.mean(predicted[:,1])

    arr.append \
      ( [ meanbiasmu
        , meanuncertmu
        ]
      )

  def savepdf(fig, name):
    return fig.savefig(f"{prefix}{name}{label}.pdf")

  fig = Figure((6, 6))
  plt = fig.add_subplot()
  plt.plot(range(1, nmax+1), [ a[0] for a in arr ], label="predicted")

  if (ntrain > 0):
    ymin, ymax = plt.get_ylim()
    plt.plot([ntrain]*2, [ymin, ymax], alpha=0.5, color="black", ls=":", label="max $\phi$ training", zorder=10)

  plt.set_xlabel(r"number of observations")
  plt.set_ylabel(r"mean bias for $\mu$")
  plt.legend()
  fig.tight_layout()
  savepdf(fig, "meanbiasmu")
  fig.clf()

  ns = np.mgrid[1:nmax+1:nmax*1j]
  fig = Figure((6, 6))
  plt = fig.add_subplot()
  plt.plot(range(1, nmax+1), [ a[1] for a in arr ], label="predicted")

  if groundtruth is not None:
    plt.plot(ns, groundtruth(ns), ls="--", color="red", label="ground truth")

  plt.set_xlabel(r"number of observations")
  plt.set_ylabel(r"mean uncertainty for $\mu$")
  plt.loglog()

  if (ntrain > 0):
    ymin, ymax = plt.get_ylim()
    plt.plot([ntrain]*2, [ymin, ymax], alpha=0.5, color="black", ls=":", label="max $\phi$ training", zorder=10)

  plt.legend()
  fig.tight_layout()
  savepdf(fig, "meanuncertmu")
  fig.clf()
