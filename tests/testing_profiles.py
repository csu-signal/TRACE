from pathlib import Path
from typing import final
from demo.base_profile import BaseProfile
from fake_camera import FakeCamera
from demo.featureModules.evaluation.eval_config import EvaluationConfig


@final
class TestDenseParaphrasingProfile(BaseProfile):
    """
    A demo profile that loads data where dense paraphrasing should be active.
    """

    def __init__(self, input_dir, output_dir, max_frames) -> None:
        eval = EvaluationConfig(
            directory=input_dir,
            asr=True,
            gesture=True,
        )
        super().__init__(output_dir=output_dir, eval_config=eval)
        self.max_frames = max_frames

    def is_done(self, frame_count, fail_count):
        return frame_count > self.max_frames

    def create_camera_device(self):  # pyright: ignore
        return FakeCamera()

    def create_audio_devices(self):
        return []
