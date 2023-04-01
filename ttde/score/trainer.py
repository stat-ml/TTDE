from functools import partial
from pathlib import Path
from typing import Optional, Any

import jax
import wandb
from flax import struct
from flax.training import checkpoints
from tqdm.auto import trange
from jax import jit, numpy as jnp

from ttde.dl_routine import batched_vmap, TensorDatasetX, KEY
from ttde.utils import Stopwatch
from ttde.dl_routine.trainer_base import TrainerBase


@struct.dataclass
class Stats:
    loss: jnp.ndarray
    ll: jnp.ndarray
    ll_median: jnp.ndarray
    tt_log_sqr_norm: jnp.ndarray
    log_int_p: jnp.ndarray
    nonpositive: jnp.ndarray


class Trainer(TrainerBase):
    def __init__(
        self,
        model,
        optim_state,
        loss_fn,
        post_processing,
        data_train: TensorDatasetX,
        data_val: TensorDatasetX,
        batch_sz: int,
        noise: float,
        work_dir: Optional[Path],
    ):
        super().__init__(work_dir)

        self.model = model
        self.optim_state = optim_state
        self.loss_fn = loss_fn
        self.post_processing = post_processing
        self.data_train = data_train
        self.data_val = data_val
        self.batch_sz = batch_sz
        self.noise = noise

    @partial(jit, static_argnums=(0,))
    def statistics(self, params, xs: jnp.ndarray) -> Stats:
        loss = self.loss_fn(self.model, params, xs, self.batch_sz)
        log_probs = batched_vmap(lambda x: self.model.apply(params, x, method=self.model.log_p), self.batch_sz)(xs)

        ll_sum = jnp.sum(jnp.where(log_probs == -jnp.inf, 0., log_probs))
        ll_number = (log_probs != -jnp.inf).sum()
        ll = ll_sum / ll_number
        ll_median = jnp.median(log_probs)

        tt_log_sqr_norm = self.model.apply(params, method=self.model.tt_log_sqr_norm)
        log_int_p = self.model.apply(params, method=self.model.log_int_p)

        nonpositive = 1 - ll_number / len(log_probs)

        return Stats(
            loss=loss,
            ll=ll,
            ll_median=ll_median,
            tt_log_sqr_norm=tt_log_sqr_norm,
            log_int_p=log_int_p,
            nonpositive=nonpositive,
        )

    @partial(jit, static_argnums=(0,))
    def train_step(self, optim_state, xs: jnp.ndarray, key=None):
        params = optim_state.target
        stats = self.statistics(params, xs)

        _, optim_state = optim_state.make_step(
            self.model, self.loss_fn, xs + jax.random.normal(key, xs.shape) * self.noise
        )

        if self.post_processing is not None:
            optim_state = optim_state.replace(target=self.post_processing(self.model, optim_state.target))

        return optim_state, stats

    def log_statistics(self, stats: Stats, tag: str, step: int):
        wandb.log(
            {
                f'{tag}/custom_loss': stats.loss.item(),
                f'{tag}/loglikelihood': stats.ll.item(),
                f'{tag}/median_ll': stats.ll_median.item(),
                f'{tag}/tt_log_sqr_norm': stats.tt_log_sqr_norm.item(),
                f'{tag}/log_int_p': stats.log_int_p.item(),
                f'{tag}/nonpositive': stats.nonpositive.item(),
            },
            step=step,
        )

    def save_checkpoint(self, step: int):
        save_checkpoint(self.cpt_dir, self.optim_state, step)

    def load_checkpoint(self, path: Path, step: int = None):
        return load_checkpoint(path, self.optim_state, step=step)

    def fit(self, key: jnp.ndarray, n_steps: int):
        with wandb.init(
            project='ttde',
            dir=self.work_dir,
        ):
            data_iterator = self.data_train.train_iterator(key, self.batch_sz)
            save_stopwatch = Stopwatch()
            val_stopwatch = Stopwatch()

            key = KEY(19)

            with trange(n_steps) as progress:
                for step in progress:
                    key, key_curr = jax.random.split(key, 2)
                    self.optim_state, stats = self.train_step(self.optim_state, next(data_iterator), key_curr)
                    if self.work_dir is not None:
                        self.log_statistics(stats, 'train', step)

                    if self.data_val is not None and val_stopwatch.passed(.5 * 60):
                        self.log_statistics(
                            self.statistics(self.optim_state.target, self.data_val.X), 'val', step
                        )
                        val_stopwatch.restart()

                    progress.set_description(f'{stats.loss:.4f}')

                    if save_stopwatch.passed(15 * 60):
                        self.save_checkpoint(step)
                        save_stopwatch.restart()

            self.save_checkpoint(step + 1)


def save_checkpoint(cpt_dir: Path, state: Any, step: int):
    checkpoints.save_checkpoint(cpt_dir, state, step)


def load_checkpoint(cpt_dir: Path, target: Optional[Any] = None, step: Optional[int] = None):
    state = checkpoints.restore_checkpoint(cpt_dir, target, step)
    return jax.tree_map(jnp.array, state)
