"""
Base interface definition
"""

from dataclasses import dataclass, field


@dataclass
class BaseInterface:
    """
    Base class all output interfaces in the demo must inherit from.
    """

    _new: bool = field(default=True, init=False, repr=False)

    def is_new(self) -> bool:
        """
        Return True if this interface contains new data.
        """
        return self._new
