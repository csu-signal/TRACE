import os
import random
from pathlib import Path
from datetime import datetime


def create_tmp_dir() -> Path:
    """
    Create a temporary directory and return
    a Path object to it. This is guaranteed
    to be a new and unique directory.
    """
    while True:
        date_name = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
        dir = Path(f"D:\\Demo\\tmp_{date_name}_{int(random.random() * 10**6)}")
        try:
            os.makedirs(dir, exist_ok=False)
            return dir
        except FileExistsError:
            pass

def create_tmp_dir_with_featureName(featureName) -> Path:
    """
    Create a temporary directory and return
    a Path object to it. This is guaranteed
    to be a new and unique directory.
    """
    while True:
        date_name = datetime.strftime(datetime.now(), "%Y-%m-%d-%H-%M-%S")
        dir = Path(f"D:\\Demo\\tmp_{str(featureName)}_{date_name}_{int(random.random() * 10**6)}")
        try:
            os.makedirs(dir, exist_ok=False)
            return dir
        except FileExistsError:
            pass
