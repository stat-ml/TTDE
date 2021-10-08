from typing import Optional

from flax import struct, linen as lenin
from jax import vmap, numpy as jnp

from ttde.dl_routine import batched_vmap


@struct.dataclass
class LLLoss:
    def __call__(self, model: lenin.Module, params, xs: jnp.ndarray, batch_sz: Optional[int] = None) -> float:
        def log_p(x):
            return model.apply(params, x, method=model.log_p)

        if batch_sz is None:
            log_ps = vmap(log_p)(xs)
        else:
            log_ps = batched_vmap(log_p, batch_sz)(xs)

        return -log_ps.mean()
