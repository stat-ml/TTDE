from pathlib import Path

import numpy as np
from jax import numpy as jnp
import flax

from ttde import utils
from ttde.dl_routine import (
    KEY_0
)
from ttde.tt import riemannian_optimizer
from ttde.tt.losses import LLLoss
