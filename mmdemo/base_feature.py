"""
Base feature definition
"""

from abc import ABC, abstractmethod

from mmdemo.base_interface import BaseInterface


# TODO: use Generic[T] for output interfaces?
class BaseFeature(ABC):
    """
    The base class all features in the demo must implement.
    """

    def __init__(self, *args) -> None:
        self._deps = []
        self._rev_deps = []
        self._register_dependencies(args)

    def _register_dependencies(self, deps: "list[BaseFeature] | tuple"):
        """
        Add other features as dependencies which are required
        to be evaluated before this feature.

        Arguments:
        deps -- a list of dependency features
        """
        assert len(self._deps) == 0, "Dependencies have already been registered"
        for d in deps:
            self._deps.append(d)
            d._rev_deps.append(self)

    @abstractmethod
    def get_output(self, *args, **kwargs) -> BaseInterface | None:
        """
        Return output of the feature. The return type must be the output
        interface to provide new data and `None` if there is no new data.

        Arguments:
        args -- list of output interfaces from dependencies in the order
                they were registered. Calling `.is_new()` on any of these
                elements will return True if the argument has not been
                sent before. It is possible that the interface will not
                contain any data before the first new data is sent.
        """
        raise NotImplementedError

    def initialize(self):
        """
        Initialize feature. This is where all the time/memory
        heavy initialization should go. Put it here instead of
        __init__ to avoid wasting resources when extra features
        exist.
        """

    def finalize(self):
        """
        Perform any necessary cleanup.
        """

    def is_done(self) -> bool:
        """
        Return True if the demo should exit. This will
        always return False if not overridden.
        """
        return False
