from dataclasses import dataclass
from typing import final

import pytest

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface
from mmdemo.demo import Demo, DemoError


@dataclass
class Interface(BaseInterface):
    data: str


@final
class Feature(BaseFeature[Interface]):
    def __init__(self, *args, name) -> None:
        super().__init__(*args)
        self.name = name
        self.initialized = False
        self.finalized = False
        self.evaluated = False

    def initialize(self):
        self.initialized = True
        self.saved = None

    def finalize(self):
        self.finalized = True

    def get_output(self, *args) -> Interface | None:
        self.evaluated = True
        self.saved = Interface(data=f"{self.name}({' '.join(i.data for i in args)})")
        return self.saved

    def is_done(self):
        return True


@pytest.fixture
def graph():
    a = Feature(name="a")
    b = Feature(name="b")
    c = Feature(a, name="c")
    d = Feature(b, name="d")
    e = Feature(a, c, name="e")
    f = Feature(b, d, e, name="f")
    g = Feature(a, b, c, d, e, f, name="g")
    h = Feature(a, c, g, f, name="h")
    i = Feature(h, name="i")

    return (a, b, c, d, e, f, g, h, i)


def test_feature_evaluation(graph):
    """
    Test that features are evaluated the correct way
    """
    a, b, c, d, e, f, g, h, i = graph

    demo = Demo(targets=[c, d])
    demo.run()
    assert isinstance(c.saved, Interface) and isinstance(d.saved, Interface)
    assert c.saved.data == "c(a())"
    assert d.saved.data == "d(b())"

    demo = Demo(targets=[i])
    demo.run()
    assert isinstance(i.saved, Interface)
    assert (
        i.saved.data
        == "i(h(a() c(a()) g(a() b() c(a()) d(b()) e(a() c(a())) f(b() d(b()) e(a() c(a())))) f(b() d(b()) e(a() c(a())))))"
    )


def test_unused_features(graph):
    """
    Test that unused features are not initialized or evaluated. Also
    check that unused features are not finalized but all others are.
    """
    a, b, c, d, e, f, g, h, i = graph

    demo = Demo(targets=[a, b, e, g])
    demo.run()

    expected = [True, True, True, True, True, True, True, False, False]

    assert [i.initialized for i in (a, b, c, d, e, f, g, h, i)] == expected
    assert [i.evaluated for i in (a, b, c, d, e, f, g, h, i)] == expected
    assert [i.finalized for i in (a, b, c, d, e, f, g, h, i)] == expected


def test_cycle_detection(graph):
    a, b, c, d, e, f, g, h, i = graph

    # make sure no error happens
    Demo(targets=[i])

    a._register_dependencies([h])

    # error should happen now
    errored = False
    try:
        Demo(targets=[i])
    except DemoError:
        errored = True

    assert errored, "Demo should have raised a DemoError exception when a cycle exists"
