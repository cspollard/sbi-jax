import jax.numpy as np
from flax import linen as nn
import jax
import distrax
from model import NMAX

PHINODES = 64
PHILAYERS = 6
RHONODES = PHINODES
RHOLAYERS = PHILAYERS

philayers = [1] + [PHINODES]*PHILAYERS
rholayers = [RHONODES]*RHOLAYERS  + [4]

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
  walk = np.repeat(np.expand_dims(np.arange(NMAX), axis=1), bsize, axis=1).T
  ns = ns.repeat(NMAX, axis=1)
  mask = walk < ns
  mask = np.repeat(np.expand_dims(mask, axis=2), philayers[-1], axis=2)

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
  dist = distrax.MultivariateNormalDiag(outs[:,:2], outs[:,2:])
  return - np.sum(dist.log_prob(pois))


def runloss(params, embed, interp, batch, ns, pois):
  outs = fwd(params, embed, interp, batch, ns)
  return loss(outs, pois)


def runlossrho(params, embed, interp, batch, ns, pois):
  outs = fwdrho(params, embed, interp, batch, ns)
  return loss(outs, pois)
