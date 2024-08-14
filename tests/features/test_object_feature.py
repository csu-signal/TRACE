from typing import final

import pytest

from mmdemo.features.objects.object_feature import Object
from mmdemo.interfaces import ColorImageInterface, ObjectInterface3D


@final
@pytest.mark.xfail
def test_import():
    """
    Check that imports work
    """
    from mmdemo.interfaces import ColorImageInterface, ObjectInterface3D


@final
@pytest.mark.xfail
def test_input_interfaces(objects: Object):
    args = objects.get_input_interfaces()
    assert len(args) == 1
    assert isinstance(args, list)


@pytest.mark.xfail
def test_output_interface(objects: Object):
    assert isinstance(objects.get_output_interface(), ObjectInterface3D)


def test_output(objects: Object):
    output = objects.get_output()
    assert isinstance(output, ObjectInterface3D)
