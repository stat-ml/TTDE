from functools import partial

from jax import numpy as jnp, vmap, jit, lax
from jax.scipy.special import logsumexp

from ttde.dl_routine import batched_vmap, nonbatched_vmap
from ttde.tt.basis import SplineOnKnots


def log_p_of_rank_1(probs, bases, x: jnp.ndarray):
    bs = vmap(type(bases).__call__)(bases, x)
    ps = (probs * bs).sum(1)

    return jnp.log(ps).sum()


def rank1_model(probs, bases, xs):
    func = lambda x: log_p_of_rank_1(probs, bases, x)
    return batched_vmap(func, 2 ** 10)(xs)


def int_of_p(probs, basis):
    return (probs * basis.integral()).sum()


@jit
def coeffs_to_valid(basis: SplineOnKnots, coeffs: jnp.ndarray) -> jnp.ndarray:
    coeffs = jnp.maximum(coeffs, 0.)
    int_p = coeffs.dot(basis.integral())
    coeffs /= jnp.where(int_p == 0., 1., int_p)

    return coeffs


@jit
def ls_abs(basis: SplineOnKnots, xs: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    A = basis.l2_integral()
    b = (batched_vmap(basis, 2 ** 10)(xs) * weights[:, None]).sum(0) / weights.sum()

    coeffs = jnp.linalg.lstsq(A, b)[0]

    return coeffs_to_valid(basis, coeffs)


def em_step_for_rank1(basis: SplineOnKnots, xs: jnp.ndarray, coeffs: jnp.ndarray, weights: jnp.ndarray):
    # E step
    #
    # p(x | i) = basis(x) / basis.integral()
    #
    # 1) p(x) = basis(x) * coeffs
    # 2) p(x) = p(x | i) * alpha = basis(x) / basis.integral() * alpha
    # (1), (2) => basis(x) * coeffs = basis(x) / basis.integral() * alpha -> alpha = coeffs * basis.integral()
    #
    # p(i | x) = p(x | i) * p(i) / p(x) = p(x | i) * alpha / p(x)
    #   = basis(x) / basis.integral * coeff * basis.integral()
    #   = basis(x) * coeff

    qs = batched_vmap(basis, 2 ** 10)(xs) * coeffs[None, :]
    qs_sum = qs.sum(1, keepdims=True)
    qs /= jnp.where(qs_sum == 0., 1., qs_sum)
    qs *= weights[:, None]

    # M step
    alpha_new = qs.sum(0) / qs.sum()
    coeffs_new = alpha_new / basis.integral()

    return coeffs_to_valid(basis, coeffs_new)


def em_for_rank_1(
        basis: SplineOnKnots, xs: jnp.ndarray, init_coeffs: jnp.ndarray, weights: jnp.ndarray, n_steps: int
):
    def body(_, coeffs):
        return em_step_for_rank1(basis, xs, coeffs, weights)

    return lax.fori_loop(0, n_steps, body, init_coeffs)


@jit
def least_squares_abs_then_em_rank1(
        basis: SplineOnKnots, xs: jnp.ndarray, weights: jnp.ndarray, n_steps: int = 10
) -> jnp.ndarray:
    init_coeffs = ls_abs(basis, xs, weights)
    return em_for_rank_1(basis, xs, init_coeffs, weights, n_steps)


def continuous_rank_1(bases: SplineOnKnots, samples: jnp.ndarray, weights: jnp.ndarray, n_steps: int = 10):
    probs = nonbatched_vmap(
        lambda basis, xs: least_squares_abs_then_em_rank1(basis, xs, weights, n_steps)
    )(bases, samples.T)
    return probs


def em_step(bases, probs, alphas, samples):
    # E
    log_qs = nonbatched_vmap(lambda p: rank1_model(p, bases, samples))(probs).T
    log_qs = log_qs + jnp.log(alphas)[None]
    log_qs -= logsumexp(log_qs, axis=1, keepdims=True)
    qs = jnp.exp(log_qs)
    #     chex.assert_tree_all_finite(qs)

    # M
    alphas_new = qs.sum(axis=0) / qs.sum()
    #     chex.assert_tree_all_finite(alphas_new)

    probs_new = batched_vmap(lambda q: continuous_rank_1(bases, samples, q), batch_sz=1)(qs.T)
    #     chex.assert_tree_all_finite(probs_new)

    return probs_new, alphas_new


@partial(jit, static_argnums=4)
def em(bases, probs, alphas, samples, n_steps: int):
    (probs, alphas), _ = lax.scan(
        lambda state, _: (em_step(bases, state[0], state[1], samples), None), (probs, alphas), jnp.arange(n_steps)
    )
    return probs, alphas
