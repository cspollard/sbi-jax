import jax.lax as lax
from einops import repeat
import jax.numpy as np
from matplotlib.figure import Figure
from utils import splitkey
from model import prior, gen, NMAX
from deepset import fwd


def plot(keyrest, batches, phi, rho, params, label=""):
  arr = []

  for n in range(1, NMAX+1):
    key , keyrest = splitkey(keyrest)

    labels = prior(key, batches)

    key , keyrest = splitkey(keyrest)

    obs , _ = gen(keyrest, labels, NMAX)

    ns = repeat(np.array([n]), "w -> h w", h=batches, w=1)

    predicted = fwd(params, phi, rho, lax.stop_gradient(obs), ns)

    meanbiasmu = np.mean(predicted[:,0] - labels[:,0])
    meanbiassig = np.mean(predicted[:,1] - labels[:,1])
    meanuncertmu = np.mean(predicted[:,2])
    meanuncertsig = np.mean(predicted[:,3])

    arr.append \
      ( [ meanbiasmu
        , meanbiassig
        , meanuncertmu
        , meanuncertsig
        ]
      )

  fig = Figure((6, 6))
  plt = fig.add_subplot()
  plt.plot(range(1, NMAX+1), [ a[0] for a in arr ])
  plt.set_xlabel(r"number of observations")
  plt.set_ylabel(r"mean bias for $\mu$")
  fig.savefig(f"meanbiasmu{label}.pdf")
  fig.clf()


  fig = Figure((6, 6))
  plt = fig.add_subplot()
  plt.plot(range(1, NMAX+1), [ a[1] for a in arr ])
  plt.set_xlabel(r"number of observations")
  plt.set_ylabel(r"mean bias for $\sigma$")
  fig.savefig(f"meanbiassigma{label}.pdf")
  fig.clf()

  fig = Figure((6, 6))
  plt = fig.add_subplot()
  plt.plot(range(1, NMAX+1), [ a[2] for a in arr ])
  plt.set_xlabel(r"number of observations")
  plt.set_ylabel(r"mean uncertainty for $\mu$")
  fig.savefig(f"meanuncertmu{label}.pdf")
  fig.clf()

  fig = Figure((6, 6))
  plt = fig.add_subplot()
  plt.plot(range(1, NMAX+1), [ a[3] for a in arr ])
  plt.set_xlabel(r"number of observations")
  plt.set_ylabel(r"mean uncertainty for $\sigma$")
  fig.savefig(f"meanuncertsigma{label}.pdf")
  fig.clf()
