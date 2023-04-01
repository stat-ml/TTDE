import jax
from flax import struct
from jax import numpy as jnp, ops, vmap

from ttde.tt.tensors import TT, TTOperator
from ttde.utils import cached_einsum


@struct.dataclass
class TTOpt:
    first: jnp.ndarray
    inner: jnp.ndarray
    last: jnp.ndarray

    @classmethod
    def zeros(cls, n_dims: int, dim: int, rank: int):
        return TTOpt(jnp.zeros([1, dim, rank]), jnp.zeros([n_dims - 2, rank, dim, rank]), jnp.zeros([rank, dim, 1]))

    @classmethod
    def from_tt(cls, tt: TT):
        return cls(tt.cores[0], jnp.stack(tt.cores[1:-1], axis=0), tt.cores[-1])

    @classmethod
    def rank_1_from_vectors(cls, vectors: jnp.ndarray):
        """
        vectors: [N_DIMS, DIM]
        """
        return cls(vectors[0, None, :, None], vectors[1:-1, None, :, None], vectors[-1, None, :, None])

    @classmethod
    def from_canonical(cls, vectors: jnp.ndarray):
        """
        vectors: [RANK, N_DIMS, DIM]
        """
        first = vectors[:, 0, :, None].T

        inner = jnp.zeros([vectors.shape[1] - 2, vectors.shape[0], vectors.shape[2], vectors.shape[0]])
        inner = inner.at[:, jnp.arange(vectors.shape[0]), :, jnp.arange(vectors.shape[0])].set(vectors[:, 1:-1, :])

        last = vectors[:, -1, :, None]

        return cls(first, inner, last)

    @property
    def n_dims(self) -> int:
        return 2 + self.inner.shape[0]

    def to_nonopt_tt(self):
        return TT([self.first, *self.inner, self.last])

    def abs(self) -> 'TTOpt':
        return TTOpt(jnp.abs(self.first), jnp.abs(self.inner), jnp.abs(self.last))


@struct.dataclass
class TTOperatorOpt:
    first: jnp.ndarray
    inner: jnp.ndarray
    last: jnp.ndarray

    @classmethod
    def from_tt_operator(cls, tt: TTOperator):
        return cls(tt.cores[0], jnp.stack(tt.cores[1:-1], axis=0), tt.cores[-1])

    @classmethod
    def rank_1_from_matrices(cls, matrices: jnp.ndarray):
        return cls(matrices[0, None, :, :, None], matrices[1:-1, None, :, :, None], matrices[-1, None, :, :, None])


@struct.dataclass
class NormalizedValue:
    value: jnp.ndarray
    log_norm: float

    @classmethod
    def from_value(cls, value):
        sqr_norm = (value ** 2).sum()
        norm_is_zero = sqr_norm == 0
        updated_sqr_norm = jnp.where(norm_is_zero, 1., sqr_norm)

        return cls(
            log_norm=jnp.where(norm_is_zero, -jnp.inf, .5 * jnp.log(updated_sqr_norm)),
            value=value / jnp.sqrt(updated_sqr_norm)
        )


def normalized_inner_product(tt1: TTOpt, tt2: TTOpt):
    def body(state, cores):
        G1, G2 = cores
        contracted = NormalizedValue.from_value(cached_einsum('ij,ikl,jkn->ln', state.value, G1, G2))
        return (
            NormalizedValue(
                value=contracted.value,
                log_norm=jnp.where(state.log_norm == -jnp.inf, -jnp.inf, state.log_norm + contracted.log_norm)
            ),
            None
        )

    state = NormalizedValue.from_value(cached_einsum('ikl,jkn->ln', tt1.first, tt2.first))
    state, _ = jax.lax.scan(body, state, (tt1.inner, tt2.inner))
    state, _ = body(state, (tt1.last, tt2.last))

    return state


def normalized_dot_operator(tt: TTOpt, tt_op: TTOperatorOpt):
    def body(x, A):
        c = jnp.einsum('rms,tmnu->rtnsu', x, A)
        return c.reshape(c.shape[0] * c.shape[1], c.shape[2], c.shape[3] * c.shape[4])

    return TTOpt(
        body(tt.first, tt_op.first),
        vmap(body)(tt.inner, tt_op.inner),
        body(tt.last, tt_op.last)
    )
