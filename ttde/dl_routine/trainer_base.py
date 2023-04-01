from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ttde.utils import PathType


@dataclass
class TrainerBase:
    work_dir: Optional[PathType]

    def __post_init__(self):
        if self.work_dir is not None:
            if not isinstance(self.work_dir, Path):
                self.work_dir = Path(self.work_dir)

            if self.work_dir.exists():
                self.work_dir.rmdir()
            self.work_dir.mkdir(parents=True)

            self.cpt_dir = self.work_dir / 'cpts'
            self.cpt_dir.mkdir(parents=True)

    def __hash__(self) -> int:
        return hash(id(self))
