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

    def __init__(self, output_dir) -> None:
        eval = EvaluationConfig(
            directory=Path(__file__).parent / "demo_inputs" / "dense_paraphrasing",
            asr=True,
            gesture=True,
        )
        super().__init__(output_dir=output_dir, eval_config=eval)

    def create_camera_device(self):  # pyright: ignore
        return FakeCamera()

    def create_audio_devices(self):
        return []
