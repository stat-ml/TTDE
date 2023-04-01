from __future__ import annotations

from typing import Any

import optax
from jax import numpy as jnp, value_and_grad
from flax import struct, linen as lenin


@struct.dataclass
class FlaxWrapper:
    optim_state: Any
    target: jnp.ndarray
    optimizer: Any = struct.field(pytree_node=False)

    @classmethod
    def create(cls, optimizer, target):
        return cls(optim_state=optimizer.init(target), target=target, optimizer=optimizer)

    def make_step(self, model: lenin.Module, loss_fn, xs: jnp.ndarray) -> FlaxWrapper:
        value, params_grad = value_and_grad(loss_fn, 1)(model, self.target, xs)
        updates, optim_state = self.optimizer.update(params_grad, self.optim_state, self.target)
        new_target = optax.apply_updates(self.target, updates)

        return value, self.replace(optim_state=optim_state, target=new_target)
