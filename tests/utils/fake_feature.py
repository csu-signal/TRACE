from typing import final

from mmdemo.base_feature import BaseFeature


@final
class FakeFeature(BaseFeature):
    """
    Fake feature that can be used for testing.

    Some features require other features to initialize,
    so this can be used in place of a real feature.
    """

    def get_output(self, *args):
        return None
