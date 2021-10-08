from jax import numpy as jnp, ops


def fuse_canonical_probs_and_alphas(probs: jnp.ndarray, alphas: jnp.ndarray):
    return ops.index_update(probs, ops.index[:, 0], probs[:, 0] * alphas[:, None])
