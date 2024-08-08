import pytest

from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces import EmptyInterface


@pytest.mark.parametrize("interface", [BaseInterface(), EmptyInterface()])
def test_is_new(interface):
    assert interface.is_new()
    interface._new = False
    assert not interface.is_new()
