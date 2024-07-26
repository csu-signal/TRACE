from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluationConfig:
    directory: str | Path
    objects: bool = False
    gesture: bool = False
    asr: bool = False
    prop: bool = False
    move: bool = False
