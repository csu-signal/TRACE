"""
Demo class definition
"""

from mmdemo import BaseFeature


class Demo:
    """
    Create a runnable demo instance

    Arguments:
        targets -- list of features which will be updated
            during the demo. The dependencies of these
            features will also be updated.
    """

    def __init__(self, *, targets: list[BaseFeature]) -> None:
        pass

    def run(self):
        pass
