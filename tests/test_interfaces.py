import pytest

from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces import EmptyInterface


@pytest.fixture(params=[(BaseInterface, ()), (EmptyInterface, ())])
def interface(request):
    cls, args = request.param
    return cls(*args)


def test_base_class(interface):
    assert isinstance(interface, BaseInterface)


def test_is_new(interface):
    assert interface.is_new()
    interface._new = False
    assert not interface.is_new()
