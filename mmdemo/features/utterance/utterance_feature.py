from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import TranscriptionInterface, UtteranceChunkInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class ASR(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return [TranscriptionInterface]

    @classmethod
    def get_output_interface(cls):
        return UtteranceChunkInterface

    def initialize(self):
        pass

    def get_output(self, tran: TranscriptionInterface):
        if not tran.is_new():
            return None

        # call __, create interface, and return

    # def select_audio_device():
    #     p = pyaudio.PyAudio()
    #     # create list of available devices
    #     print("Available devices:")
    #     for i in range(p.get_device_count()):
    #         print(i, ":", p.get_device_info_by_index(i).get('name'))
    #     # select device
    #     device_index = int(input("Select device index: "))
    #     print("Selected device:", p.get_device_info_by_index(device_index).get('name'))
    #     p.terminate()
    #     return device_index

    # def build_utterances(
    #     builder_queue: "mp.Queue[AsrDeviceData]",
    #     processor_queue: mp.Queue,
    #     use_vad=True,
    #     max_utterance_time=10,
    #     output_dir = None
    # ):
    #     if output_dir is None:
    #         output_dir = Path(".")
    #     else:
    #         output_dir = Path(output_dir)

    #     os.makedirs(output_dir / "chunks", exist_ok=True)

    #     stored_audio = defaultdict(bytes)
    #     starts = defaultdict(float)
    #     contains_activity = defaultdict(bool)
    #     total_time = defaultdict(float)

    #     if use_vad:
    #         vad = load_silero_vad()
    #     else:
    #         vad = None

    #     counter = 0

    #     while True:
    #         data = builder_queue.get()
    #         if data is None:
    #             break
    #         id, start, stop, frames, sample_rate, sample_width, channels = (
    #             data.id,
    #             data.start_time,
    #             data.stop_time,
    #             data.frames,
    #             data.sample_rate,
    #             data.sample_width,
    #             data.channels
    #         )

    #         wf = wave.open(str(output_dir / "chunks" / "vad_tmp.wav"), 'wb')
    #         wf.setnchannels(channels)
    #         wf.setsampwidth(sample_width)
    #         wf.setframerate(sample_rate)
    #         wf.writeframes(frames)
    #         wf.close()

    #         if use_vad:
    #             try:
    #                 audio = read_audio(str(output_dir / "chunks" / "vad_tmp.wav"))
    #                 activity = len(get_speech_timestamps(audio, vad)) > 0
    #             except RuntimeError:
    #                 activity = False
    #         else:
    #             activity = True

    #         if not activity and not contains_activity[id]:
    #             starts[id] = start
    #             stored_audio[id] = frames
    #             total_time[id] = stop - start
    #         else:
    #             if len(stored_audio[id]) == 0:
    #                 starts[id] = start
    #             stored_audio[id] += frames
    #             total_time[id] += stop - start

    #         if activity:
    #             contains_activity[id] = True

    #         # if there is no activity but there was previous activity, make utterance
    #         if (not activity and contains_activity[id]) or total_time[id] > max_utterance_time:
    #             next_file = str(output_dir / "chunks" / f"{counter:08}.wav")
    #             wf = wave.open(next_file, 'wb')
    #             wf.setnchannels(channels)
    #             wf.setsampwidth(sample_width)
    #             wf.setframerate(sample_rate)
    #             wf.writeframes(stored_audio[id])
    #             wf.close()

    #             processor_queue.put(AsrUtteranceData(id, starts[id], stop, next_file))

    #             stored_audio[id] = b''
    #             contains_activity[id] = False
    #             total_time[id] = 0
    #             counter += 1

    #     processor_queue.put(None)

    # def process_utterances(queue: "mp.Queue[AsrUtteranceData]", print_output=False, output_queue=None):
    #     # model = faster_whisper.WhisperModel("large-v2", compute_type="float16")
    #     model = faster_whisper.WhisperModel("small", compute_type="float16")

    #     while True:
    #         data = queue.get()
    #         if data is None:
    #             break
    #         name, start_time, stop_time, chunk_file = data.id, data.start_time, data.stop_time, data.audio_file

    #         segments, info = model.transcribe(chunk_file, language="en")
    #         transcription = " ".join(segment.text for segment in segments if segment.no_speech_prob < 0.5)  # Join segments into a single string

    #         if print_output:
    #             print(f'{name}: {transcription}')

    #         if output_queue is not None:
    #             output_queue.put((name, start_time, stop_time, transcription, chunk_file))

    #     if output_queue is not None:
    #         output_queue.put(None)
