import pytest

from mmdemo.utils.frame_time_converter import FrameTimeConverter


@pytest.mark.parametrize(
    "data,ft_expected,tf_expected",
    [
        (
            [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
            [(0, 0), (2, 2), (4, 4)],
            [(0, 0), (3, 3), (4, 4)],
        ),
        (
            [(0, 5), (2, 6), (8, 20), (13, 21), (19, 22)],
            [(0, 5), (19, 22), (7, 6), (10, 20), (20, 22)],
            [(5, 0), (22, 19), (10, 2), (50, 19)],
        ),
    ],
)
def test_frame_time_converter(data, ft_expected, tf_expected):
    converter = FrameTimeConverter()
    for frame, time in data:
        converter.add_data(frame, time)

    for f, t in ft_expected:
        assert converter.get_time(f) == t

    for t, f in tf_expected:
        assert converter.get_frame(t) == f

    assert converter.get_num_datapoints() == len(data)
