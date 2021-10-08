from __future__ import annotations

from flax import struct

from jax import numpy as jnp, vmap, lax

from ttde.utils import tree_stack
from ttde.utils.polynomial import poly_shift, poly_x, poly_definite_int, Polynomial#, poly_int


def poly_const(const: float) -> jnp.ndarray:
    return jnp.array([const])


@struct.dataclass
class PolyPiece:
    l: float
    r: float
    poly: Polynomial

    def __call__(self, x: float):
        return jnp.where((self.l <= x) & (x < self.r), jnp.polyval(self.poly, x), 0.)

    def integral(self) -> float:
        return poly_definite_int(self.poly, self.l, self.r)

    def integral_up_to(self, x: float) -> float:
        up_to = jnp.minimum(x, self.r)
        return jnp.where(self.l < x, poly_definite_int(self.poly, self.l, up_to), 0.)

    @classmethod
    def l2_of_product(cls, lhs: PolyPiece, rhs: PolyPiece) -> float:
        L = jnp.maximum(lhs.l, rhs.l)
        R = jnp.minimum(lhs.r, rhs.r)

        return jnp.where(L < R, poly_definite_int(jnp.polymul(lhs.poly, rhs.poly), L, R), 0.)

    @classmethod
    def l2_up_to_of_product(cls, lhs: PolyPiece, rhs: PolyPiece, x: float) -> float:
        L = jnp.maximum(lhs.l, rhs.l)
        R = jnp.minimum(lhs.r, jnp.minimum(rhs.r, x))

        return jnp.where(L < R, poly_definite_int(jnp.polymul(lhs.poly, rhs.poly), L, R), 0.)
#
#
@struct.dataclass
class PiecewisePoly:
    pieces: PolyPiece  # batch

    def __call__(self, x: float):
        return vmap(PolyPiece.__call__, in_axes=(0, None))(self.pieces, x).sum()

    def integral(self):
        return vmap(PolyPiece.integral)(self.pieces).sum()

    def integral_up_to(self, x: float) -> float:
        return vmap(PolyPiece.integral_up_to, in_axes=(0, None))(self.pieces, x).sum()

    @classmethod
    def l2_of_product(cls, lhs: PiecewisePoly, rhs: PiecewisePoly):
        l2_with_each_other_func = vmap(vmap(PolyPiece.l2_of_product, in_axes=(0, None)), in_axes=(None, 0))
        return l2_with_each_other_func(lhs.pieces, rhs.pieces).sum()

    @classmethod
    def l2_up_to(cls, lhs: PiecewisePoly, rhs: PiecewisePoly, x: float):
        l2_up_to_with_each_other_func = vmap(vmap(
            PolyPiece.l2_up_to_of_product,
            in_axes=(0, None, None)),
            in_axes=(None, 0, None)
        )
        return l2_up_to_with_each_other_func(lhs.pieces, rhs.pieces, x).sum()


def build_spline_on_knots(knots: jnp.ndarray) -> PiecewisePoly:
    q = len(knots) - 2

    x = poly_x()

    def w(i: int, k: int):
        return poly_shift(x, knots[i]) / (knots[i + k] - knots[i])

    b_prev = [
        [
            PolyPiece(knots[j], knots[j + 1], poly_const(1.) if i == j else poly_const(0.))
            for j in range(q + 1)
        ]
        for i in range(q + 1)
    ]

    for k in range(1, q + 1):
        b_next = []
        for i in range(q - k + 1):
            left, right = b_prev[i: i + 2]
            w_left = w(i, k)
            w_right = jnp.polysub(poly_const(1.), w(i + 1, k))

            ith_b = []
            for piece in range(q + 1):
                poly = jnp.polyadd(jnp.polymul(left[piece].poly, w_left), jnp.polymul(right[piece].poly, w_right))
                ith_b.append(PolyPiece(left[piece].l, left[piece].r, poly))
            b_next.append(ith_b)

        b_prev = b_next

    return PiecewisePoly(tree_stack(b_prev[0]))


@struct.dataclass
class SplineOnKnots:
    splines: PiecewisePoly  # batch
    knots: jnp.ndarray
    q: int = struct.field(pytree_node=False)

    @classmethod
    def from_uniform_knots(cls, l: float, r: float, n: int, q: int) -> SplineOnKnots:
        h = (r - l) / (n - q)
        return cls.from_knots(q, jnp.linspace(l - h * q, r + h * q, n + q + 1))

    @classmethod
    def from_knots(cls, q: int, knots: jnp.ndarray) -> SplineOnKnots:
        start_inds = jnp.arange(len(knots) - q - 1)[:, None]
        batch_of_knots = vmap(lax.dynamic_slice, in_axes=(None, 0, None))(knots, start_inds, [q + 2])
        return SplineOnKnots(
            splines=vmap(build_spline_on_knots)(batch_of_knots),
            knots=knots,
            q=q,
        )

    @property
    def dim(self):
        return len(self.knots) - self.q - 1

    @property
    def left_zero_bound(self):
        return self.knots[0]

    @property
    def right_zero_bound(self):
        return self.knots[-1]

    def __call__(self, x):
        return jnp.abs(vmap(PiecewisePoly.__call__, in_axes=(0, None))(self.splines, x))

    def integral(self) -> jnp.ndarray:
        return vmap(PiecewisePoly.integral)(self.splines)

    def integral_up_to(self, x: float) -> jnp.ndarray:
        return vmap(PiecewisePoly.integral_up_to, in_axes=(0, None))(self.splines, x)

    def l2_integral(self) -> jnp.ndarray:
        l2_with_each_other_func = vmap(vmap(PiecewisePoly.l2_of_product, in_axes=(0, None)), in_axes=(None, 0))
        return l2_with_each_other_func(self.splines, self.splines)

    def l2_up_to(self, x: float):
        return vmap(vmap(PiecewisePoly.l2_up_to, in_axes=(0, None, None)), in_axes=(None, 0, None))(
            self.splines, self.splines, x
        )


def create_space_uniform_knots(xs: jnp.ndarray, n: int, q: int):
    l, r = xs.min(), xs.max()
    h = (r - l) / (n - q)
    return jnp.linspace(l - h * q, r + h * q, n + q + 1)
