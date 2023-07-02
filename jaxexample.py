import jax.numpy as np
import jax
from jax import random
from flax import linen as nn
import distrax
import optax

NMAX = 128
BATCHSIZE = 128
EPOCHSIZE = 1024
NEPOCHS = 2048*16

philayers = [1] + [32]*6
rholayers = [32]*6  + [6]

k = random.PRNGKey(0)

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

phiparams = phi.init(k, np.zeros((1, 13, 1)))
rhoparams = rho.init(k, phi.apply(phiparams, np.zeros((1, 1, 1))))

modelparams = { "rho" : rhoparams , "phi" : phiparams }

def masksum(ten, ns):
  walk = np.repeat(np.expand_dims(np.arange(NMAX), axis=1), BATCHSIZE, axis=1).T
  ns = ns.repeat(NMAX, axis=1)
  mask = walk < ns
  mask = np.repeat(np.expand_dims(mask, axis=2), philayers[-1], axis=2)

  return np.sum(ten, axis=1, where=mask)


def loss(params, embed, interp, pois, batch, ns):
  embedding = embed.apply(params["phi"], batch)
  summed = masksum(embedding, ns)
  outs = interp.apply(params["rho"], summed)
  dist = distrax.MultivariateNormalDiag(outs[:,:3], np.exp(outs[:,3:]))
  pois = np.stack(pois, axis=1).squeeze()
  return - np.sum(dist.log_prob(pois))


def gen(key, params, nmax):
  lam, mu, sig = params
  batches = lam.shape[0]
  ns = random.poisson(key, lam, dtype=int)
  ten = random.normal(key, (batches, nmax, 1))
  ten = np.expand_dims(mu, 2) + np.expand_dims(sig, 2) * ten
  return ten , ns


def prior(key, batches):
  lam = random.gamma(key, 64, (batches, 1))
  mu = random.normal(key, (batches, 1))
  sig = random.gamma(key, 50, (batches, 1))
  return lam , mu , sig


optimizer = optax.adam(learning_rate=1e-4)
opt_state = optimizer.init(modelparams)

@jax.jit
def step(params, opt_state, batch, ns, labels):
  loss_value, grads = jax.value_and_grad(loss)(params, phi, rho, labels, batch, ns)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss_value


for _ in range(100):
  for _ in range(EPOCHSIZE):
    labels = prior(k, BATCHSIZE)
    batch , ns = gen(k, labels, NMAX)
    modelparams, opt_state, loss_value = \
      step(modelparams, opt_state, batch, ns, labels)


  print(loss(modelparams, phi, rho, labels, batch, ns))
