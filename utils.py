import jax.numpy as np
from jax import random

def splitkey(key):
  s = random.split(key)
  return s[0] , s[1]
