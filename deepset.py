from einops import repeat, rearrange
import jax.numpy as np
from flax import linen as nn
import jax
import distrax

PHINODES = 128
PHILAYERS = 3
RHONODES = PHINODES
RHOLAYERS = PHILAYERS

philayers = [1] + [PHINODES]*PHILAYERS
rholayers = [RHONODES]*RHOLAYERS  + [2]

phi = \
  nn.Sequential \
  ( [ layer
      for size in philayers
      for layer in [ nn.Conv(size, [1]) , nn.relu ]
    ][:-1]
    + [ nn.softmax ]
  )

rho = \
  nn.Sequential \
  ( [ layer for size in rholayers for layer in [ nn.Dense(size) , nn.relu ] ][:-1]
  )


def masksum(ten, ns):
  bsize = ns.shape[0]
  nmax = ten.shape[1]
  walk = repeat(np.arange(0, nmax), "w -> b w", b = bsize)

  mask = walk < ns
  mask = repeat(mask, "b w -> b w 128")
  return np.sum(ten, axis=1, where=mask)

def fwd(params, embed, interp, batch, ns):
  embedding = embed.apply(params["phi"], batch)
  summed = masksum(embedding, ns)
  outs = interp.apply(params["rho"], summed)
  outs = outs.at[:,1:].set(np.exp(outs[:,1:]))
  return outs

def fwdrho(params, embed, interp, batch, ns):
  embedding = embed.apply(params["phi"], batch)
  summed = jax.lax.stop_gradient(masksum(embedding, ns))
  outs = interp.apply(params["rho"], summed)
  outs = outs.at[:,1:].set(np.exp(outs[:,1:]))
  return outs

def loss(outs, pois):
  dist = distrax.MultivariateNormalDiag(outs[:,:1], outs[:,1:])
  return - np.sum(dist.log_prob(pois))

def runloss(params, embed, interp, batch, ns, pois):
  outs = fwd(params, embed, interp, batch, ns)
  return loss(outs, pois)

def runlossrho(params, embed, interp, batch, ns, pois):
  outs = fwdrho(params, embed, interp, batch, ns)
  return loss(outs, pois)
