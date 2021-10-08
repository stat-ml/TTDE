from jax import numpy as jnp


Polynomial = jnp.ndarray


def poly_x() -> Polynomial:
    return jnp.array([1, 0])


def poly_int(coeffs: Polynomial) -> Polynomial:
    return jnp.concatenate([coeffs / jnp.arange(len(coeffs), 0, -1), jnp.zeros(1)])


def poly_definite_int(coeffs: Polynomial, l: float, r: float) -> float:
    integral = poly_int(coeffs)
    return jnp.polyval(integral, r) - jnp.polyval(integral, l)


def poly_shift(p: Polynomial, h: float) -> Polynomial:
    """
    p(x) -> p(x - h)
    """

    res = jnp.zeros([1])

    x_m_h = jnp.array([1, -h])
    x_m_h_p = jnp.ones([1])

    for i in range(len(p)):
        res = jnp.polyadd(res, x_m_h_p * p[-i - 1])
        x_m_h_p = jnp.polymul(x_m_h_p, x_m_h)

    return res
