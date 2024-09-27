from enum import Enum, auto


class DeviceType(Enum):
    """
    An enum representing which type of azure kinect
    device should be opened.
    """

    CAMERA = auto()
    PLAYBACK = auto()
