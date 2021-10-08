import jax
from jax import vmap, numpy as jnp
from jax.scipy.special import logsumexp

from ttde.dl_routine import MutableModule, repeat
from ttde.score.models.continuous_canonical_init import continuous_rank_1, int_of_p, em
from ttde.score.models.discrete_canonical_init import fuse_canonical_probs_and_alphas
from ttde.tt.basis import SplineOnKnots
from ttde.tt.tt_opt import TTOpt, normalized_inner_product, TTOperatorOpt, normalized_dot_operator
from ttde.utils import index


class PAsTTOptBase(MutableModule):
    bases: SplineOnKnots = None
    permutations: jnp.ndarray = None
    rank: int = None

    @classmethod
    def create(
        cls,
        key: jnp.ndarray,
        bases: SplineOnKnots,
        n_components: int,
        rank: int,
    ):
        assert n_components >= 1
        n_dims = len(bases.knots)

        print('creating permutations...')
        permutations = [jnp.arange(n_dims)]
        perm_keys = jax.random.split(key, n_components - 1)
        for key in perm_keys:
            permutations.append(jax.random.permutation(key, n_dims))
        permutations = jnp.array(permutations)

        print(f'creating {cls.__name__}...')
        return cls(bases=bases, permutations=permutations, rank=rank)

    @property
    def n_components(self):
        return self.permutations.shape[0]

    @property
    def n_dims(self):
        return self.bases.knots.shape[0]

    def p(self, x):
        return jnp.exp(self.log_p(x))

    def log_p(self, x, eps=-jnp.inf):
        return self.unnormalized_log_p(x, eps) - self.log_int_p()

    def setup(self):
        self.tt = self.variable(
            'tt', 'tt',
            repeat(TTOpt.zeros, self.n_components),
            self.n_dims, index(self.bases)[0].dim, self.rank
        )

    def change_tt(self, tt: TTOpt):
        self.tt.value = tt

    def __call__(self):
        pass

    def tt_log_sqr_norm(self):
        def one_tt_sqr_norm(tt):
            return normalized_inner_product(tt, tt).log_norm

        return logsumexp(vmap(one_tt_sqr_norm)(self.tt.value))

    def normalize_cores(self):
        def one_normalize(tt):
            def one_core_normalize(core):
                return core / jnp.linalg.norm(core)
            return TTOpt(
                first=one_core_normalize(tt.first),
                inner=vmap(one_core_normalize)(tt.inner),
                last=one_core_normalize(tt.last),
            )
        self.tt.value = vmap(one_normalize)(self.tt.value)

    def init_components_from_one_canonical(self, canonical: jnp.ndarray):
        """
        canonical: [rank, n_dims, basis_dim]
        """

        canonicals = canonical[:, self.permutations, :]
        canonicals = jnp.moveaxis(canonicals, 1, 0)

        tt = vmap(TTOpt.from_canonical)(canonicals)

        self.change_tt(tt)

    def add_noise(self, key: jnp.ndarray, noise: float):
        key_first, key_inner, key_last = jax.random.split(key, 3)
        tt = self.tt.value
        self.change_tt(
            TTOpt(
                tt.first + jax.random.normal(key_first, tt.first.shape) * noise,
                tt.inner + jax.random.normal(key_inner, tt.inner.shape) * noise,
                tt.last + jax.random.normal(key_last, tt.last.shape) * noise,
            )
        )

    def init_rank_1(self, key: jnp.ndarray, samples: jnp.ndarray, noise: float = 1e-2):
        print('rank-1...')
        rank1 = continuous_rank_1(self.bases, samples, jnp.ones(len(samples)))
        canonical = jnp.pad(rank1[None], [(0, self.rank - 1), (0, 0), (0, 0)])

        self.init_components_from_one_canonical(canonical)

    def init_canonical(self, key: jnp.ndarray, samples: jnp.ndarray, n_steps: int):
        print('rank1 init for em...')
        rank1_probs = continuous_rank_1(self.bases, samples, jnp.ones(len(samples)), 10)

        noise_level = 0.1
        repeated_probs = jnp.repeat(rank1_probs[None], self.rank, 0)
        noise_tensor = jax.random.uniform(key, repeated_probs.shape)
        noised_probs = repeated_probs * (1 - noise_level + noise_tensor * noise_level * 2)
        noised_probs /= vmap(vmap(int_of_p), in_axes=(0, None))(noised_probs, self.bases)[..., None]

        init_probs = noised_probs
        init_alphas = jnp.ones(self.rank) / self.rank

        print('em...')
        probs, alphas = em(self.bases, init_probs, init_alphas, samples, n_steps)
        fused_probs = fuse_canonical_probs_and_alphas(probs, alphas)

        self.init_components_from_one_canonical(fused_probs)


class PAsTTSqrOpt(PAsTTOptBase):
    def unnormalized_log_p(self, x, eps=-jnp.inf):
        def one_log_p(bs, perm, tt):
            bs = bs[perm]
            b_tensor = TTOpt.rank_1_from_vectors(bs)
            normalized = normalized_inner_product(tt, b_tensor)
            return jnp.where(normalized.log_norm == -jnp.inf, eps, 2 * normalized.log_norm)

        bs = vmap(type(self.bases).__call__)(self.bases, x)

        log_ps = vmap(one_log_p, in_axes=(None, 0, 0))(bs, self.permutations, self.tt.value)

        return logsumexp(log_ps)

    def log_int_p(self):
        def one_log_int_p(Ds, perm, tt):
            Ds = Ds[perm]
            D_operator = TTOperatorOpt.rank_1_from_matrices(Ds)
            return normalized_inner_product(tt, normalized_dot_operator(tt, D_operator)).log_norm

        Ds = vmap(type(self.bases).l2_integral)(self.bases)

        log_int_ps = vmap(one_log_int_p, in_axes=(None, 0, 0))(Ds, self.permutations, self.tt.value)

        return logsumexp(log_int_ps)

    def fix_nonsqrt_init(self):
        tt = self.tt.value
        self.change_tt(TTOpt(jnp.sqrt(tt.first), jnp.sqrt(tt.inner), jnp.sqrt(tt.last)))

    def init_rank_1(self, key: jnp.ndarray, samples: jnp.ndarray, noise: float = 1e-2):
        super().init_rank_1(key, samples, noise)
        self.fix_nonsqrt_init()

    def init_canonical(self, key: jnp.ndarray, samples: jnp.ndarray, n_steps: int):
        super().init_canonical(key, samples, n_steps)
        self.fix_nonsqrt_init()
