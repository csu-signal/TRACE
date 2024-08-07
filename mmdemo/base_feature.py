"""
Base feature definition
"""

from abc import ABC, abstractmethod
from typing import Type, final

from mmdemo.base_interface import BaseInterface


class BaseFeature(ABC):
    """
    The base class all features in the demo must implement.
    """

    def __init__(self) -> None:
        self._deps = []
        self._rev_deps = []

    @final
    def register_dependencies(
        self, interfaces: list[Type[BaseInterface]], deps: "list[BaseFeature] | tuple"
    ):
        """
        Add other features as dependencies which are required
        to be evaluated before this feature.

        Arguments:
        interfaces -- a list of required dependency interface types
        deps -- a list of dependency features
        """
        assert len(self._deps) == 0, "Dependencies have already been registered"
        assert len(deps) == len(
            interfaces
        ), f"{len(interfaces)} input features were expected but {len(deps)} were provided."

        for d, i_type in zip(deps, interfaces):
            assert isinstance(
                d.get_output_interface(), i_type
            ), f"A dependency did not have output interface {i_type}"

            self._deps.append(d)
            d._rev_deps.append(self)

    @classmethod
    @abstractmethod
    def get_output_interface(cls) -> Type[BaseInterface]:
        """
        Returns the output interface class (must be a
        subclass of BaseInterface).
        """
        raise NotImplementedError

    @abstractmethod
    def get_output(self, *args, **kwargs) -> BaseInterface | None:
        """
        Return output of the feature. The return type must be
        `self.get_output_interface()` to provide new data and
        `None` if there is no new data.

        Arguments:
        args -- list of output interfaces from dependencies in the order
                they were registered. Calling `.is_new()` on any of these
                elements will return True if the argument has not been
                sent before.
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

    def is_done(self):
        """
        Return True if the demo should exit. This will
        always return False if not overridden.
        """
        return False
