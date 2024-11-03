import jax.numpy as np
import jax
from jax import random
from flax.training import orbax_utils, train_state
import optax
from tqdm import tqdm
import orbax
from os import mkdirs

from model import gen, prior, groundtruth
from deepset import phi, rho, fwd, runloss, runlossrho
from utils import splitkey
from plot import plot


BATCHSIZE = 128
NBATCHES = 512*8
NEPOCHS = 16
LR = 1e-3
FINETUNELR = 1e-4
NPLOTPOINTS = 2000

NMAXSTART = 8
NMAXFINETUNE = 256


knext = random.PRNGKey(0)

mkdirs("initial", exist_ok=True)
mkdirs("finetuned", exist_ok=True)
mkdirs("direct", exist_ok=True)


@jax.jit
def step(params, opt_state, batch, ns, labels):
  loss_value, grads = jax.value_and_grad(runloss)(params, phi, rho, batch, ns, labels)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss_value


@jax.jit
def steprho(params, opt_state, batch, ns, labels):
  loss_value, grads = jax.value_and_grad(runlossrho)(params, phi, rho, batch, ns, labels)
  updates, opt_state = optimizer.update(grads, opt_state, params)
  params = optax.apply_updates(params, updates)
  return params, opt_state, loss_value


k, knext = splitkey(knext)
phiparams = phi.init(k, np.zeros((1, 13, 1)))
k, knext = splitkey(knext)
rhoparams = rho.init(k, phi.apply(phiparams, np.zeros((1, 1, 1))))

modelparams = { "rho" : rhoparams , "phi" : phiparams }

sched = optax.cosine_decay_schedule(LR , NEPOCHS*NBATCHES)
optimizer = optax.adam(learning_rate=sched)
opt_state = optimizer.init(modelparams)

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()

for iepoch in range(NEPOCHS):
  print("epoch:", iepoch)
  for _ in tqdm(range(NBATCHES)):
    k, knext = splitkey(knext)
    labels = prior(k, BATCHSIZE)
    k, knext = splitkey(knext)
    batch , ns = gen(k, labels, NMAXSTART)
    modelparams, opt_state, loss_value = \
      step(modelparams, opt_state, batch, ns, labels)


  k, knext = splitkey(knext)
  plot(knext, NPLOTPOINTS, phi, rho, modelparams, NMAXFINETUNE, prefix="initial/", label=f"_epoch{iepoch:02d}", ntrain=NMAXSTART, groundtruth=groundtruth)

  save_args = orbax_utils.save_args_from_target(modelparams)

  orbax_checkpointer.save("/Users/cspollard/Physics/sbi-jax/initial.orbax", modelparams, save_args=save_args, force=True)


sched = optax.cosine_decay_schedule(FINETUNELR , NEPOCHS*NBATCHES)
optimizer = optax.adam(learning_rate=sched)
opt_state = optimizer.init(rhoparams)

for iepoch in range(NEPOCHS):
  print("epoch:", iepoch)
  for _ in tqdm(range(NBATCHES)):
    k, knext = splitkey(knext)
    labels = prior(k, BATCHSIZE)
    k, knext = splitkey(knext)
    batch , ns = gen(k, labels, NMAXFINETUNE)
    rhoparams, opt_state, loss_value = \
      steprho(rhoparams, opt_state, batch, ns, labels)


  k, knext = splitkey(knext)
  plot \
    ( knext
    , NPLOTPOINTS
    , phi
    , rho
    , { "phi" : modelparams["phi"] , "rho" : rhoparams }
    , NMAXFINETUNE
    , prefix="finetuned/"
    , label=f"_epoch{iepoch:02d}"
    , ntrain=NMAXSTART
    , groundtruth=groundtruth
    )

  save_args = orbax_utils.save_args_from_target(modelparams)

  orbax_checkpointer.save \
    ( "/Users/cspollard/Physics/sbi-jax/finetuned.orbax"
    , { "phi" : modelparams["phi"] , "rho" : rhoparams }
    , save_args=save_args
    , force=True
    )


k, knext = splitkey(knext)
phiparams = phi.init(k, np.zeros((1, 13, 1)))
k, knext = splitkey(knext)
rhoparams = rho.init(k, phi.apply(phiparams, np.zeros((1, 1, 1))))

modelparams = { "rho" : rhoparams , "phi" : phiparams }

sched = optax.cosine_decay_schedule(LR , NEPOCHS*NBATCHES)
optimizer = optax.adam(learning_rate=sched)
opt_state = optimizer.init(modelparams)

for iepoch in range(NEPOCHS):
  print("epoch:", iepoch)
  for _ in tqdm(range(NBATCHES)):
    k, knext = splitkey(knext)
    labels = prior(k, BATCHSIZE)
    k, knext = splitkey(knext)
    batch , ns = gen(k, labels, NMAXFINETUNE)
    modelparams, opt_state, loss_value = \
      steprho(modelparams, opt_state, batch, ns, labels)


  k, knext = splitkey(knext)
  plot \
    ( knext
    , NPLOTPOINTS
    , phi
    , rho
    , modelparams
    , NMAXFINETUNE
    , prefix="direct/"
    , label=f"_epoch{iepoch:02d}"
    , ntrain=NMAXSTART
    , groundtruth=groundtruth
    )

  save_args = orbax_utils.save_args_from_target(modelparams)

  orbax_checkpointer.save \
    ( "/Users/cspollard/Physics/sbi-jax/direct.orbax"
    , modelparams
    , save_args=save_args
    , force=True
    )
