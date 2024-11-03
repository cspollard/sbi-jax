from einops import repeat
import jax.numpy as np
from jax import random
from utils import splitkey

SIGMAMU = 20
MAXSIGMA = 10
NMAXINITIAL = 2
NMAXFINETUNE = 16


def gen(keyrest, params, nmax):
  batches = params.shape[0]
  mu = params
  key , keyrest = splitkey(keyrest)
  sigma = random.uniform(key, (batches, 1)) * MAXSIGMA

  key , keyrest = splitkey(keyrest)
  ns = random.randint(key, (batches, 1), 1, nmax+1, dtype=int)

  key , keyrest = splitkey(keyrest)
  ten = \
    random.normal(key, (batches, nmax, 1)) \
    * repeat(sigma, "b 1 -> b m 1", m=nmax)

  ten = repeat(mu, "b w -> b m w", m=nmax) + ten
  return ten , ns


def prior(keyrest, batches):
  key , keyrest = splitkey(keyrest)
  mu = random.normal(key, (batches, 1)) * SIGMAMU
  return mu

groundtruth = None
# def groundtruth(ns):
#   return np.sqrt(1.0 / (1.0 / SIGMAMU**2 + ns / SIGMAX**2))
