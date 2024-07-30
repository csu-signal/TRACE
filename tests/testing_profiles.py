

from typing import final
from demo.base_profile import BaseProfile
from fake_camera import FakeCamera
from demo.featureModules.evaluation.eval_config import EvaluationConfig


@final
class TestDenseParaphrasingProfile(BaseProfile):
    def __init__(self) -> None:
        super().__init__(eval_config=EvaluationConfig(directory="test_inputs\\dense_paraphrasing", asr=True, gesture=True))

    def create_camera_device(self): #pyright: ignore
        return FakeCamera()

    def create_audio_devices(self):
        return []
