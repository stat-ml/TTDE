from dataclasses import dataclass

from ttde.score.experiment_setups.base import Base


@dataclass
class Trainer(Base):
    batch_sz: int
    noise: float
    lr: float
