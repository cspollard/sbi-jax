import jax.numpy as np
from jax import random

def splitkey(key):
  s = random.split(key)
  return s[0] , s[1]

def dmap(f, d):
  return { k : f(k, d[k]) for k in d }

def dmap1(f, d):
  return { k : f(d[k]) for k in d }
