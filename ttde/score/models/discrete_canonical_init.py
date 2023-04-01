from jax import numpy as jnp


def fuse_canonical_probs_and_alphas(probs: jnp.ndarray, alphas: jnp.ndarray):
    return probs.at[:, 0].multiply(alphas[:, None])
