from einops import repeat
import jax.numpy as np
from jax import random
from utils import splitkey

SIGMAX = 10
SIGMAMU = 20

def gen(keyrest, params, nmax):
  batches = params.shape[0]
  mu = params

  key , keyrest = splitkey(keyrest)
  ns = random.randint(key, (batches, 1), 0, nmax, dtype=int)
  key , keyrest = splitkey(keyrest)
  ten = random.normal(key, (batches, nmax, 1)) * SIGMAX

  ten = repeat(mu, "b w -> b m w", m=nmax) + ten
  return ten , ns


def prior(keyrest, batches):
  key , keyrest = splitkey(keyrest)
  mu = random.normal(key, (batches, 1)) * SIGMAMU
  # key , keyrest = splitkey(keyrest)
  # sig = random.gamma(key, 50, (batches, 1))
  return mu

def groundtruth(ns):
  return np.sqrt(1.0 / (1.0 / SIGMAMU**2 + ns / SIGMAX**2))
