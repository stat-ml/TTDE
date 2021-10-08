from dataclasses import dataclass
from typing import Type

from jax import numpy as jnp, vmap

from ttde.score.models import opt_for_table_data
from ttde.score.experiment_setups import base
from ttde.tt.basis import SplineOnKnots, create_space_uniform_knots


@dataclass
class PAsTTOptBase(base.Base):
    q: int
    m: int
    rank: int
    n_comps: int

    @staticmethod
    def one_basis(m: int, q: int, xs: jnp.ndarray):
        return SplineOnKnots.from_knots(q, create_space_uniform_knots(xs, m, q))

    def bases(self, samples: jnp.ndarray):
        return vmap(self.one_basis, in_axes=(None, None, 1))(self.m, self.q, samples)

    def _create(self, key: jnp.ndarray, samples: jnp.ndarray, model_cls: Type[opt_for_table_data.PAsTTOptBase]):
        print('initializing bases...')
        bases = self.bases(samples)
        print('initializing model...')
        model = model_cls.create(key, bases, self.n_comps, self.rank)
        return model


@dataclass
class PAsTTSqrOpt(PAsTTOptBase):
    def create(self, key: jnp.ndarray, samples: jnp.ndarray):
        return self._create(key, samples, opt_for_table_data.PAsTTSqrOpt)

    postprocessing = None
