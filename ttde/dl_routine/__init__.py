from dataclasses import dataclass
from typing import Generator, Callable, TypeVar

import jax
from jax import numpy as jnp, vmap, lax
from flax import linen as lenin


@dataclass
class TensorDatasetX:
    X: jnp.ndarray

    def __len__(self):
        return len(self.X)

    def to_jax(self):
        return TensorDatasetX(jnp.array(self.X))

    def train_iterator(self, key: jnp.ndarray, batch_sz: int) -> Generator[jnp.ndarray, None, None]:
        while True:
            key, curr_key = jax.random.split(key, 2)
            inds = jax.random.randint(curr_key, [batch_sz], 0, len(self.X))
            yield self.X[inds]

    def reshape(self, *shape) -> 'TensorDatasetX':
        return TensorDatasetX(self.X.reshape(self.X.shape[0], *shape))


def repeat(func, n_times):
    def stub(_, *args):
        return func(*args)

    def wrapper(*args):
        return vmap(stub, in_axes=[0] + [None] * len(args))(jnp.arange(n_times), *args)

    return wrapper


F = TypeVar("F", bound=Callable)


def batched_vmap(f: F, batch_sz: int) -> F:
    def wrapper(*xs):
        n_batches = xs[0].shape[0] // batch_sz

        def body_from_f(_, xs):
            return None, vmap(f)(*xs)

        batched = jax.lax.scan(
            body_from_f,
            None,
            [x[:n_batches * batch_sz].reshape(n_batches, batch_sz, *x.shape[1:]) for x in xs],
        )[1]
        batched = batched.reshape(n_batches * batch_sz, *batched.shape[2:])

        remainder = vmap(f)(*(x[n_batches * batch_sz:] for x in xs))

        return jnp.concatenate([batched, remainder], 0)

    return wrapper


def nonbatched_vmap(func):
    def wrapper(*args):
        def body(_, arg):
            return None, func(*arg)

        return lax.scan(body, None, args)[1]

    return wrapper


KEY_0 = jax.random.PRNGKey(0)


def KEY(i: int) -> jnp.ndarray:
    return jax.random.PRNGKey(i)


class MutableModule(lenin.Module):
    def mutate(self, variables, *args, rngs=None, method=None, **kwargs):
        return self.apply(variables, *args, rngs=rngs, method=method, mutable=True, **kwargs)[1]
