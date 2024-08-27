import os
import shutil
import wave
from collections import defaultdict, deque
from typing import final

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import AudioFileInterface, AudioFileListInterface
from mmdemo.utils.files import create_tmp_dir


@final
class VADUtteranceBuilder(BaseFeature[AudioFileInterface]):
    """
    Turn audio input chunks into utterances using VAD segmentation.
    This will attempt to leave one chunk with no voice activity
    on each side of the utterance.

    Input interfaces are any number `AudioFileListInterface`.
    Each input feature must stay consistent in the format of
    input (rate, channels, etc.).

    Output interface is `AudioFileInterface`. If the frame rate is too
    slow the output of this feature may fall behind, but this is unlikely
    since utterances don't happen very often.

    Keyword arguments:
    `delete_input_files` -- True if input audio files should be deleted, default False
    `max_utterance_time` -- the maximum number of seconds an utterance can be or None, default 5
    """

    def __init__(
        self,
        *args: BaseFeature[AudioFileListInterface],
        delete_input_files=False,
        max_utterance_time: float | None = 5,
    ):
        super().__init__(*args)
        self.delete_input_files = delete_input_files
        self.max_utterance_time = max_utterance_time

    def initialize(self):
        self.counter = 0

        self.vad = load_silero_vad()

        # store current state of utterances for each speaker
        self.current_data = defaultdict(bytes)
        self.contains_activity = defaultdict(bool)
        self.starts = defaultdict(float)
        self.total_time = defaultdict(float)

        self.output_dir = create_tmp_dir()

        self.outputs = deque()

    def finalize(self):
        shutil.rmtree(self.output_dir)

    def get_output(self, *args: AudioFileListInterface) -> AudioFileInterface | None:
        for audio_input_list in args:
            if not audio_input_list.is_new():
                continue

            for audio_input in audio_input_list.audio_files:
                # run input through vad
                audio = read_audio(str(audio_input.path))
                activity = len(get_speech_timestamps(audio, self.vad)) > 0

                # load frames and params from file
                wave_reader = wave.open(str(audio_input.path), "rb")
                chunk_n_frames = wave_reader.getnframes()
                chunk_frames = b""
                for _ in range(chunk_n_frames // 1024 + 1):
                    chunk_frames += wave_reader.readframes(1024)
                params = wave_reader.getparams()
                wave_reader.close()

                if self.delete_input_files:
                    os.remove(audio_input.path)

                if activity:
                    if len(self.current_data[audio_input.speaker_id]) == 0:
                        # if no data has been stored yet, set the start time
                        self.starts[audio_input.speaker_id] = audio_input.start_time
                        self.total_time[audio_input.speaker_id] = 0

                    # add audio to the stored frames and update params
                    self.current_data[audio_input.speaker_id] += chunk_frames
                    self.total_time[audio_input.speaker_id] += (
                        audio_input.end_time - audio_input.start_time
                    )
                    self.contains_activity[audio_input.speaker_id] = True
                else:
                    if self.contains_activity[audio_input.speaker_id]:
                        # if we have stored activity, create a new utterance
                        self.current_data[audio_input.speaker_id] += chunk_frames
                        self.total_time[audio_input.speaker_id] += (
                            audio_input.end_time - audio_input.start_time
                        )
                        self.create_utterance(audio_input.speaker_id, params)

                    # reset to only storing the last chunk
                    self.starts[audio_input.speaker_id] = audio_input.start_time
                    self.current_data[audio_input.speaker_id] = chunk_frames
                    self.total_time[audio_input.speaker_id] = (
                        audio_input.end_time - audio_input.start_time
                    )
                    self.contains_activity[audio_input.speaker_id] = False

                # force output file to be created if the time is too long.
                # this will only happen if there has been activity, otherwise
                # an utterance would have just been created
                if (
                    self.max_utterance_time is not None
                    and self.total_time[audio_input.speaker_id]
                    >= self.max_utterance_time
                ):
                    # create utterance and remove all data
                    self.create_utterance(audio_input.speaker_id, params)
                    self.current_data[audio_input.speaker_id] = b""
                    self.contains_activity[audio_input.speaker_id] = False

        # return one output at a time
        if len(self.outputs) > 0:
            return self.outputs.popleft()

        return None

    def create_utterance(self, speaker_id, params):
        """
        Create an utterance file based on saved data and add to `self.outputs`
        """
        next_file = self.output_dir / f"{self.counter:08}.wav"
        wf = wave.open(str(next_file), "wb")
        wf.setparams(params)
        wf.writeframes(self.current_data[speaker_id])
        wf.close()

        self.outputs.append(
            AudioFileInterface(
                speaker_id=speaker_id,
                start_time=self.starts[speaker_id],
                end_time=self.starts[speaker_id] + self.total_time[speaker_id],
                path=next_file,
            )
        )

        self.counter += 1
