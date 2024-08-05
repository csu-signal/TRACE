from abc import ABC, abstractmethod
from typing import Type, final

from mmdemo import BaseInterface


class BaseFeature(ABC):
    """
    The base class all features in the demo
    must implement.
    """

    def __init__(self) -> None:
        self._deps = []
        self._rev_deps = []

        self._clean = False

    @final
    def register_dependencies(self, deps: "list[BaseFeature]"):
        """
        Add other features as dependencies which are required
        to be evaluated before this feature.

        Arguments:
        deps -- a list of dependency features
        """
        for d in deps:
            self._deps.append(d)
            d._rev_deps.append(self)

    @final
    def _mark_dirty(self):
        if self.should_mark_dirty():
            self._clean = False
            for d in self._rev_deps:
                d._mark_dirty()

    @abstractmethod
    @classmethod
    def get_output_interface(cls) -> Type[BaseInterface]:
        """
        Returns the output interface class (must be a
        subclass of BaseInterface).
        """
        raise NotImplementedError

    @abstractmethod
    def get_output(self, *args: BaseInterface) -> BaseInterface:
        """
        Return output of the feature. The return type must be
        `self.get_output_interface()`.

        Arguments:
        args -- list of output interfaces from dependencies in the order
                they were registered
        """
        raise NotImplementedError

    def initialize(self):
        """
        Initialize feature. This is where all the time/memory
        heavy initialization should go. Put it here instead of
        __init__ to avoid wasting resources when extra features
        exist.
        """
        pass

    def finalize(self):
        """
        Perform any necessary cleanup.
        """
        pass

    def is_done(self):
        """
        Return True if the demo should exit. This will
        always return False if not overridden.
        """
        return False

    def should_mark_dirty(self):
        """
        Return True if the feature should be marked as dirty
        (meaning it needs to be re-evaluated). This will always
        return True if not overridden and is called whenever a
        dependency is marked as dirty.
        """
        return True

    def has_external_input(self):
        """
        Return True if the feature should be marked as dirty
        (meaning it needs to be re-evaluated) because of some
        input not related to its dependencies. This will always
        return False if not overridden as will be called
        periodically during the demo.
        """
        return False
