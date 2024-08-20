from typing import final

import pytest

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface


class Interface(BaseInterface):
    count: int
    data: tuple


@final
class Feature(BaseFeature[Interface]):
    def get_output(self, *args) -> Interface | None:
        return None


@final
class Input(BaseFeature[Interface]):
    pass
