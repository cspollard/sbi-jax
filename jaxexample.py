import jax.numpy as np
import jax
from jax import random
from flax import linen as nn
import distrax
import optax

NMAX = 128
BATCHSIZE = 128
EPOCHSIZE = 512
NEPOCHS = 2048*16

philayers = [1] + [32]*6
rholayers = [32]*6  + [6]

knext = random.PRNGKey(0)

def splitkey(key):
  s = random.split(key)
  return s[0] , s[1]


phi = \
  nn.Sequential \
  ( [ layer
      for size in philayers
      for layer in [ nn.Conv(size, [1]) , nn.relu ]
    ][:-1]
    + [ lambda x: nn.softmax(x, axis=2) ]
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
  return outs


def loss(outs, pois):
  dist = distrax.MultivariateNormalDiag(outs[:,:3], np.exp(outs[:,3:]))
  return - np.sum(dist.log_prob(pois))


def runloss(params, embed, interp, batch, ns, pois):
  outs = fwd(params, embed, interp, batch, ns)
  return loss(outs, pois)



def gen(keyrest, params, nmax):
  lam = params[:,0:1]
  mu = params[:,1:2]
  sig = params[:,2:3]
  batches = lam.shape[0]

  key , keyrest = splitkey(keyrest)
  ns = random.poisson(key, lam, dtype=int)
  key , keyrest = splitkey(keyrest)
  ten = random.normal(key, (batches, nmax, 1))

  ten = np.expand_dims(mu, 2) + np.expand_dims(sig, 2) * ten
  return ten , ns


def prior(keyrest, batches):
  key , keyrest = splitkey(keyrest)
  lam = random.gamma(key, 64, (batches, 1))
  key , keyrest = splitkey(keyrest)
  mu = random.normal(key, (batches, 1))
  key , keyrest = splitkey(keyrest)
  sig = random.gamma(key, 50, (batches, 1))
  return np.stack([lam, mu, sig], axis=1).squeeze()


@jax.jit
def step(params, opt_state, batch, ns, labels):
  loss_value, grads = jax.value_and_grad(runloss)(params, phi, rho, batch, ns, labels)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss_value


k, knext = splitkey(knext)
phiparams = phi.init(k, np.zeros((1, 13, 1)))
k, knext = splitkey(knext)
rhoparams = rho.init(k, phi.apply(phiparams, np.zeros((1, 1, 1))))

modelparams = { "rho" : rhoparams , "phi" : phiparams }

optimizer = optax.adam(learning_rate=1e-4)
opt_state = optimizer.init(modelparams)


k, knext = splitkey(knext)
testlabels = prior(k, 1000)
testbatch , testns = gen(k, testlabels, NMAX)

for _ in range(NEPOCHS):
  for _ in range(EPOCHSIZE):
    k, knext = splitkey(knext)
    labels = prior(k, BATCHSIZE)
    k, knext = splitkey(knext)
    batch , ns = gen(k, labels, NMAX)
    modelparams, opt_state, loss_value = \
      step(modelparams, opt_state, batch, ns, labels)


  outs = fwd(modelparams, phi, rho, testbatch, testns)
  print(loss(outs, testlabels))
  print()
  print(outs[:2,:3] - testlabels[:2])
  print()

