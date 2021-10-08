from __future__ import annotations
from jax import numpy as jnp, value_and_grad, grad, hessian
from flax import struct, linen as lenin, optim, traverse_util


@struct.dataclass
class FlaxWrapper:
    optim_state: optim.OptimizerDef
    target: jnp.ndarray

    @classmethod
    def create(cls, flax_optim: optim.OptimizerDef, target, focus=None):
        return cls(optim_state=flax_optim.create(target=target, focus=focus), target=target)

    def make_step(self, model: lenin.Module, loss_fn, xs: jnp.ndarray) -> FlaxWrapper:
        optim_state = self.optim_state.replace(target=self.target)

        params = optim_state.target
        value, params_grad = value_and_grad(loss_fn, 1)(model, params, xs)

        optim_state = optim_state.apply_gradient(params_grad)

        return value, self.replace(optim_state=optim_state, target=optim_state.target)
