import jax.numpy as np
from jax import random
from utils import splitkey

NMAX = 64

def gen(keyrest, params, nmax):
  batches = params.shape[0]
  mu = params[:,0:1]
  sig = params[:,1:2]

  key , keyrest = splitkey(keyrest)
  ns = random.randint(key, (batches, 1), 0, nmax, dtype=int)
  key , keyrest = splitkey(keyrest)
  ten = random.normal(key, (batches, nmax, 1))

  ten = np.expand_dims(mu, 2) + np.expand_dims(sig, 2) * ten
  return ten , ns


def prior(keyrest, batches):
  key , keyrest = splitkey(keyrest)
  mu = random.normal(key, (batches, 1))
  key , keyrest = splitkey(keyrest)
  sig = random.gamma(key, 50, (batches, 1))
  return np.stack([mu, sig], axis=1).squeeze()
