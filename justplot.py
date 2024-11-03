import jax.numpy as np
import jax.random as random
import orbax.checkpoint

from model import groundtruth, NMAXINITIAL, NMAXFINETUNE
from deepset import phi, rho
from utils import splitkey
from plot import plot

knext = random.PRNGKey(0)

k, knext = splitkey(knext)
phiparams = phi.init(k, np.zeros((1, 13, 1)))
k, knext = splitkey(knext)
rhoparams = rho.init(k, phi.apply(phiparams, np.zeros((1, 1, 1))))

modelparams = { "rho" : rhoparams , "phi" : phiparams }

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
initialparams = orbax_checkpointer.restore("/Users/cspollard/Physics/sbi-jax/initial.orbax", modelparams)
finetunedparams = orbax_checkpointer.restore("/Users/cspollard/Physics/sbi-jax/finetuned.orbax", modelparams)
directparams = orbax_checkpointer.restore("/Users/cspollard/Physics/sbi-jax/direct.orbax", modelparams)

dparams = \
  { "initial" : initialparams
  , "finetuned" : finetunedparams
  , "direct" : directparams
  }

plot(knext, 100, phi, rho, dparams, NMAXFINETUNE, label="_compare", ntrain=NMAXINITIAL, groundtruth=groundtruth)
