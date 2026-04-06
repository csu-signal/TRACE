"""
Speech output for sensor-sheet friction responses.
"""
from __future__ import annotations

from pathlib import Path

from kokoro import KPipeline
import sounddevice as sd
import torch

try:
    import winsound
except ImportError:
    winsound = None

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import SensorSheetFrictionOutputInterface, SpeechOutputInterface

ENABLE_TTS = True


def extract_speakable_friction_text(friction_statement: str) -> str:
    return friction_statement.strip()


def play_notification_bell() -> None:
    if winsound is not None:
        winsound.MessageBeep(winsound.MB_ICONASTERISK)
        return

    sample_rate = 24000
    duration_seconds = 0.18
    t = torch.arange(int(sample_rate * duration_seconds), dtype=torch.float32) / sample_rate
    tone = 0.2 * torch.sin(2 * torch.pi * 880 * t)
    sd.play(tone.numpy(), sample_rate)
    sd.wait()


class SensorSpeechOutput(BaseFeature[SpeechOutputInterface]):
    """
    Speak the latest friction response generated for the sensor demo.
    """

    def __init__(self, friction: BaseFeature[SensorSheetFrictionOutputInterface]):
        super().__init__(friction)
        self.speechoutput = False
        self.length = 0

    def initialize(self):
        self.pipeline = KPipeline(lang_code="a")
        voice_path = Path(__file__).with_name("am_michael.pt")
        self.voice_tensor = torch.load(voice_path, weights_only=True)
        self.last_friction = ""

    def get_output(self, frictionout: SensorSheetFrictionOutputInterface):
        friction = extract_speakable_friction_text(frictionout.friction_statement)

        if not friction or friction == self.last_friction or self.length > -30:
            self.length -= 1
            return SpeechOutputInterface(speech_output=self.speechoutput, length=self.length)

        if not ENABLE_TTS:
            self.last_friction = friction
            self.speechoutput = False
            return SpeechOutputInterface(speech_output=self.speechoutput, length=self.length)

        play_notification_bell()

        generator = self.pipeline(
            friction,
            voice=self.voice_tensor,
            speed=1,
            split_pattern=r"\n+",
        )

        self.speechoutput = False
        for _, _, audio in generator:
            sd.play(audio, 24000)
            sd.wait()
            self.length = 30
            self.speechoutput = True

        self.last_friction = friction
        return SpeechOutputInterface(speech_output=self.speechoutput, length=self.length)