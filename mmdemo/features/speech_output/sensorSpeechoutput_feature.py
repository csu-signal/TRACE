"""
Speech output for sensor-sheet friction responses.
"""
from __future__ import annotations

from pathlib import Path

from kokoro import KPipeline
import queue
import threading

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
        self.tts_cooldown_counter = 0
        self.tts_queue = queue.Queue()
        self.tts_thread = None
        self.last_friction = ""
        self.is_speaking = False

    def initialize(self):
        self.pipeline = KPipeline(lang_code="a")
        voice_path = Path(__file__).with_name("am_michael.pt")
        self.voice_tensor = torch.load(voice_path, weights_only=True)
        # Start the TTS background thread
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()

    def _tts_worker(self):
        """Background worker thread that handles TTS generation and playback."""
        while True:
            try:
                friction = self.tts_queue.get(timeout=1)
                if friction is None:  # Sentinel value to stop thread
                    break

                audio_queue = queue.Queue()

                def build_audio_stream() -> None:
                    try:
                        generator = self.pipeline(
                            friction,
                            voice=self.voice_tensor,
                            speed=1,
                            split_pattern=r"\n+",
                        )
                        for _, _, audio in generator:
                            audio_queue.put(audio)
                    finally:
                        audio_queue.put(None)

                audio_thread = threading.Thread(target=build_audio_stream, daemon=True)
                audio_thread.start()

                play_notification_bell()

                while True:
                    audio = audio_queue.get()
                    if audio is None:
                        break
                    sd.play(audio, 24000)
                    sd.wait()

                self.is_speaking = False
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[SensorSpeechOutput._tts_worker] ERROR: {e}")
                self.is_speaking = False

    def get_output(self, frictionout: SensorSheetFrictionOutputInterface):
        friction = extract_speakable_friction_text(frictionout.friction_statement)

        if not friction or friction == self.last_friction or self.tts_cooldown_counter > -30:
            self.tts_cooldown_counter -= 1
            return SpeechOutputInterface(speech_output=self.is_speaking, length=self.tts_cooldown_counter)

        if not ENABLE_TTS:
            self.last_friction = friction
            self.speechoutput = False
            return SpeechOutputInterface(speech_output=False, length=self.tts_cooldown_counter)

        # Queue the friction text for TTS processing in the background thread
        self.is_speaking = True
        self.tts_cooldown_counter = 30
        self.last_friction = friction
        self.tts_queue.put(friction)
        return SpeechOutputInterface(speech_output=True, length=self.tts_cooldown_counter)
