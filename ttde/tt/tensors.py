from __future__ import annotations

from typing import Sequence, List

import jax
from jax import numpy as jnp
from flax import struct


@struct.dataclass
class TT:
    @classmethod
    def zeros(cls, dims: Sequence[int], rs: Sequence[int]) -> TT:
        assert len(dims) == len(rs) + 1

        rs = [1] + list(rs) + [1]
        cores = [jnp.zeros((rs[i], dim, rs[i + 1])) for i, dim in enumerate(dims)]

        return cls(cores)

    @classmethod
    def generate_random(cls, key: jnp.ndarray, dims: Sequence[int], rs: Sequence[int]) -> TT:
        assert len(dims) == len(rs) + 1

        rs = [1] + list(rs) + [1]
        keys = jax.random.split(key, len(dims))
        cores = [jax.random.normal(key, (rs[i], dim, rs[i + 1])) for i, (dim, key) in enumerate(zip(dims, keys))]

        return cls(cores)

    cores: List[jnp.ndarray]

    @property
    def n_dims(self):
        return len(self.cores)

    @property
    def full_tensor(self) -> jnp.ndarray:
        res = self.cores[0]
        for core in self.cores[1:]:
            res = jnp.einsum('...r,riR->...iR', res, core)
        return jnp.squeeze(res, (0, -1))

    def reverse(self) -> TT:
        return TT([transpose_core(core) for core in self.cores[::-1]])

    def astype(self, dtype: jnp.dtype) -> TT:
        return TT([core.astype(dtype) for core in self.cores])

    def __sub__(self, other: TT):
        return subtract(self, other)


@struct.dataclass
class TTOperator:
    @classmethod
    def generate_random(
        cls, key: jnp.ndarray, dims_from: Sequence[int], dims_to: Sequence[int], rs: Sequence[int]
    ) -> TTOperator:
        n_dims = len(dims_from)

        assert len(dims_from) == n_dims
        assert len(dims_to) == n_dims
        assert len(rs) + 1 == n_dims

        rs = [1] + list(rs) + [1]
        keys = jax.random.split(key, n_dims)
        cores = [
            jax.random.normal(key, (rs[i], dim_from, dim_to, rs[i + 1]))
            for i, (dim_from, dim_to, key) in enumerate(zip(dims_from, dims_to, keys))
        ]

        return cls(cores)

    cores: List[jnp.ndarray]

    @property
    def full_operator(self) -> jnp.ndarray:
        res = self.cores[0]
        for core in self.cores[1:]:
            res = jnp.einsum('...r,rijR->...ijR', res, core)
        return jnp.squeeze(res, (0, -1))

    def reverse(self):
        # idk, what should I do with axes 1 and 2.
        return TTOperator([jnp.moveaxis(core, (0, 1, 2, 3), (3, 1, 2, 0)) for core in self.cores[::-1]])


def transpose_core(core: jnp.ndarray) -> jnp.ndarray:
    return jnp.moveaxis(core, (0, 1, 2), (2, 1, 0))


def subtract(lhs: TT, rhs: TT) -> TT:
    assert lhs.n_dims == rhs.n_dims

    if lhs.n_dims == 1:
        return TT([lhs.cores[0] - rhs.cores[0]])

    first = jnp.concatenate([lhs.cores[0], -rhs.cores[0]], axis=-1)
    last = jnp.concatenate([lhs.cores[-1], rhs.cores[-1]], axis=0)
    inner = [
        jnp.concatenate(
            [
                jnp.concatenate([c1, jnp.zeros((c1.shape[0], c1.shape[1], c2.shape[2]))], axis=-1),
                jnp.concatenate([jnp.zeros((c2.shape[0], c2.shape[1], c1.shape[2])), c2], axis=-1),
            ],
            axis=0,
        ) for c1, c2 in zip(lhs.cores[1:-1], rhs.cores[1:-1])
    ]

    return TT([first] + inner + [last])
