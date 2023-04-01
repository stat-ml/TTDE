import time
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Union

import jax
from opt_einsum import contract_expression
from jax import numpy as jnp
from jax.tree_util import tree_flatten


def suffix_with_date(folder: Path) -> Path:
    if type(folder) is not Path:
        folder = Path(folder)
    now = str(datetime.now()).replace(' ', '_')
    return folder / now


PathType = Union[str, Path]


def cached_einsum(expr: str, *args):
    return cached_einsum_expr(expr, *[arg.shape for arg in args])(*args, backend='jax')


@lru_cache
def cached_einsum_expr(expr: str, *shapes):
    return contract_expression(expr, *shapes)


def tree_stack(trees):
    """Takes a list of trees and stacks every corresponding leaf.
    For example, given two trees ((a, b), c) and ((a', b'), c'), returns
    ((stack(a, a'), stack(b, b')), stack(c, c')).
    Useful for turning a list of objects into something you can feed to a
    vmapped function.
    """
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


@dataclass
class Stopwatch:
    last_time: float = None

    def __post_init__(self):
        self.restart()

    def restart(self):
        self.last_time = time.time()

    def get_time(self) -> float:
        return time.time() - self.last_time

    def passed(self, duration: float) -> bool:
        return self.get_time() >= duration


def index(batched_struct):
    class Indexer:
        def __getitem__(self, idx):
            return jax.tree_map(lambda x: x[idx], batched_struct)

    return Indexer()
