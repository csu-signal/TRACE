from typing import final

import pytest

from mmdemo.features.objects.object_feature import Object
from mmdemo.interfaces import ColorImageInterface, ObjectInterface3D


def test_output(objects: Object):
    output = objects.get_output()
    assert isinstance(output, ObjectInterface3D)
