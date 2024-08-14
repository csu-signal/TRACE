from pathlib import Path
from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import AudioFileInterface


@final
class MicAudio(BaseFeature[AudioFileInterface]):
    def __init__(self, *, device_id: int) -> None:
        super().__init__()
        self.device_id = device_id

    def initialize(self):
        pass

    def finalize(self):
        pass

    def get_output(self) -> AudioFileInterface | None:
        pass


@final
class RecordedAudio(BaseFeature[AudioFileInterface]):
    def __init__(self, *, path: Path) -> None:
        super().__init__()
        self.path = path

    def initialize(self):
        pass

    def finalize(self):
        pass

    def get_output(self) -> AudioFileInterface | None:
        pass
