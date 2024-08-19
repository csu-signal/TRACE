from typing import final
from mmdemo.base_feature import BaseFeature


@final
class FakeFeature(BaseFeature):
    def get_output(self, *args):
        return None
