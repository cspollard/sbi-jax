from models.maf import MaskedAutoregressiveFlow
import jax.numpy as numpy
import jax.random as random
import jax
import optax
import jax.scipy.optimize as optimize


def gaussMixture(k, logits, mus, covs):
  thisk , thatk = random.split(k)
  cs = random.categorical(thisk, logits=logits)
  return random.multivariate_normal(thatk , mus[cs] , covs[cs])


model = \
  MaskedAutoregressiveFlow \
  ( n_dim=1
  , hidden_dims=[ 64 , 64 , 64 , 64 ]
  )

def loss(params, batch):
  return - numpy.mean(model.apply(params, batch))


@jax.jit
def step(params, opt_state, batch):
  loss_value, grads = jax.value_and_grad(loss)(params, batch)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss_value


thisk , thatk = random.split(random.PRNGKey(0))
tracer = random.uniform(key=thisk, shape=(32, 1))
params = model.init(thatk, tracer)
initparams = params

optimizer = optax.adam(learning_rate=1e-4)
opt_state = optimizer.init(params)

n = 1024
logits = numpy.array([1, 1, 3], dtype=float).reshape(1, 3).repeat(n, axis=0)
mus = numpy.array([0, 1, 2], dtype=float).reshape(3, 1)
covs = numpy.array([1, 2, 3], dtype=float).reshape(3, 1, 1)


for _ in range(1000):
  thisk , thatk = random.split(thatk)
  batch = gaussMixture(thisk, logits, mus, covs)

  params , opt_state , loss_value = step(params, opt_state, batch)

import matplotlib.figure as figure

fig = figure.Figure((8, 8))
plt = fig.add_subplot()
plt.hist \
  ( [ model.apply(params, 1024, thatk, method=model.sample)[:,0]
    , batch[:,0]
    , model.apply(initparams, 1024, thatk, method=model.sample)[:,0]
    ]
  , label=["flow" , "orig" , "initial"]
  )

plt.legend()
fig.savefig("test.pdf")
