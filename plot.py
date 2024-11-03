import jax.lax as lax
from einops import repeat
import jax.numpy as np
from matplotlib.figure import Figure
from utils import splitkey
from model import prior, gen, NMAX, SIGMAMU, SIGMAX
from deepset import fwd

def groundtruth(ns):
  return np.sqrt(1.0 / (1.0 / SIGMAMU**2 + ns / SIGMAX**2))


def plot(keyrest, batches, phi, rho, params, prefix="", label=""):
  arr = []

  for n in range(1, NMAX+1):
    key , keyrest = splitkey(keyrest)

    labels = prior(key, batches)

    key , keyrest = splitkey(keyrest)

    obs , _ = gen(keyrest, labels, NMAX)

    ns = repeat(np.array([n]), "w -> h w", h=batches, w=1)

    predicted = fwd(params, phi, rho, lax.stop_gradient(obs), ns)

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
  plt.plot(range(1, NMAX+1), [ a[0] for a in arr ], label="predicted")
  plt.set_xlabel(r"number of observations")
  plt.set_ylabel(r"mean bias for $\mu$")
  plt.legend()
  fig.tight_layout()
  savepdf(fig, "meanbiasmu")
  fig.clf()

  ns = np.mgrid[1:NMAX+1:NMAX*1j]
  fig = Figure((6, 6))
  plt = fig.add_subplot()
  plt.plot(range(1, NMAX+1), [ a[1] for a in arr ], label="predicted")
  plt.plot(ns, groundtruth(ns), ls="-", color="red", label="ground truth")
  plt.set_xlabel(r"number of observations")
  plt.set_ylabel(r"mean uncertainty for $\mu$")
  plt.loglog()
  plt.legend()
  fig.tight_layout()
  savepdf(fig, "meanuncertmu")
  fig.clf()
