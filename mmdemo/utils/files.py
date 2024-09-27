import os
import random
from pathlib import Path


def create_tmp_dir() -> Path:
    """
    Create a temporary directory and return
    a Path object to it. This is guaranteed
    to be a new and unique directory.
    """
    while True:
        dir = Path(f"tmp_{int(random.random() * 10**6)}")
        try:
            os.makedirs(dir, exist_ok=False)
            return dir
        except FileExistsError:
            pass
