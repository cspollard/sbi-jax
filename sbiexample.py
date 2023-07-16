from models.maf import MaskedAutoregressiveFlow
import jax.numpy as numpy
import jax.random as random
import jax
import optax
import jax.scipy.optimize as optimize

BATCHSIZE = 1024

def gaussMixture(k, logits, mus, covs):
  thisk , thatk = random.split(k)
  cs = random.categorical(thisk, logits=logits)
  return random.multivariate_normal(thatk , mus[cs] , covs[cs])


def loss(params, conditions, batch):
  return - numpy.mean(model.apply(params, batch, conditions))


@jax.jit
def step(params, conditions, opt_state, batch):
  loss_value, grads = jax.value_and_grad(loss)(params, conditions, batch)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss_value

model = \
  MaskedAutoregressiveFlow \
  ( n_dim=1
  , hidden_dims=[64, 64, 64, 64]
  , n_context=3
  )


thisk , thatk = random.split(random.PRNGKey(0))
tracer = random.uniform(key=thisk, shape=(BATCHSIZE, 1))
logits = \
  numpy.array([0, 0, 0], dtype=float).reshape(1, 3).repeat(BATCHSIZE, axis=0)


print(jax.nn.softmax(logits))

thisk , thatk = random.split(thatk)
params = model.init(thisk, tracer, logits)
initparams = params

optimizer = optax.adam(learning_rate=1e-4)
opt_state = optimizer.init(params)

mus = numpy.array([0, 1, 2], dtype=float).reshape(3, 1)
covs = numpy.array([1, 2, 3], dtype=float).reshape(3, 1, 1)


def train(key, nepochs, params, opt_state):
  thatk = key
  for _ in range(nepochs):
    thisk , thatk = random.split(thatk)
    logits = random.uniform(thisk, shape=(BATCHSIZE, 3), minval=-10, maxval=10)

    thisk , thatk = random.split(thatk)
    batch = gaussMixture(thatk, logits, mus, covs)

    params , opt_state , loss_value = step(params, logits, opt_state, batch)

  return params, opt_state


thisk , thatk = random.split(thatk)
params, opt_state = train(thisk, 10000, initparams, opt_state)

import matplotlib.figure as figure

logits = \
  numpy.array([0, 1, 2], dtype=float).reshape(1, 3).repeat(16*BATCHSIZE, axis=0)

fig = figure.Figure((8, 8))
plt = fig.add_subplot()
plt.hist \
  ( [ model.apply(params, 16*BATCHSIZE, thatk, logits, method=model.sample)[:,0]
    , gaussMixture(thatk, logits, mus, covs)[:,0]
    , model.apply(initparams, 16*BATCHSIZE, thatk, logits, method=model.sample)[:,0]
    ]
  , label=["flow" , "orig" , "initial"]
  )

plt.legend()
fig.savefig("test.pdf")


def objective(conditions, data):
  conds = conditions.reshape((1, 3)).repeat(data.shape[0], axis=0)
  return - numpy.mean(model.apply(params, data, conds))

testdata = gaussMixture(thatk, logits, mus, covs)

initconditions = numpy.array([0, 0, 0], dtype=float)

res = \
  optimize.minimize \
  ( objective
  , initconditions
  , args=(testdata,)
  , method="BFGS"
  )

print(res.x)
print(jax.nn.softmax(res.x))
print(res.hess_inv)
