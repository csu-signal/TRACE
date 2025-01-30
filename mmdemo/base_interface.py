"""
Base interface definition
"""

from dataclasses import dataclass, field


@dataclass
class BaseInterface:
    """
    Base class all output interfaces in the demo must inherit from.
    """

    #field(default = True, init = False, repr = False) means that the default value is True, _new is global, it cannot be directly called
    _new: bool = field(default=True, init=False, repr=False)

    def is_new(self) -> bool:
        """
        Return True if this interface contains new data.
        """
        return self._new
