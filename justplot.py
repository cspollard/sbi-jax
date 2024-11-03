import jax.numpy as np
import jax.random as random
from flax.training import orbax_utils, train_state
import orbax

from model import groundtruth
from deepset import phi, rho
from utils import splitkey
from plot import plot

knext = random.PRNGKey(0)

k, knext = splitkey(knext)
phiparams = phi.init(k, np.zeros((1, 13, 1)))
k, knext = splitkey(knext)
rhoparams = rho.init(k, phi.apply(phiparams, np.zeros((1, 1, 1))))

modelparams = { "rho" : rhoparams , "phi" : phiparams }
save_args = orbax_utils.save_args_from_target(modelparams)

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
params = orbax_checkpointer.restore("/Users/cspollard/Physics/sbi-jax/state.orbax", modelparams)
finetuneparams = orbax_checkpointer.restore("/Users/cspollard/Physics/sbi-jax/finetuned.orbax", modelparams)

plot(knext, 100, phi, rho, params, 256, label="nofinetune", ntrain=8, groundtruth=groundtruth)
plot(knext, 100, phi, rho, finetuneparams, 256, label="finetune", ntrain=8, groundtruth=groundtruth)
