import multiprocessing as mp
import os
from pathlib import Path
import wave
from collections import defaultdict
from ctypes import c_bool
from dataclasses import dataclass

import faster_whisper
import pyaudio
from silero_vad import get_speech_timestamps, load_silero_vad, read_audio

from featureModules.asr.device import AsrDeviceData, BaseDevice
from featureModules.IFeature import *
from logger import Logger
from utils import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

@dataclass
class AsrUtteranceData:
    id: str
    start: float
    stop: float
    audio_file: str

def select_audio_device():
    p = pyaudio.PyAudio()
    # create list of available devices
    print("Available devices:")
    for i in range(p.get_device_count()):
        print(i, ":", p.get_device_info_by_index(i).get('name'))
    # select device
    device_index = int(input("Select device index: "))
    print("Selected device:", p.get_device_info_by_index(device_index).get('name'))
    p.terminate()
    return device_index

def build_utterances(
    builder_queue: "mp.Queue[AsrDeviceData]",
    processor_queue: mp.Queue,
    done,
    use_vad=True,
    max_utterance_time=10,
    output_dir = None
):
    if output_dir is None:
        output_dir = Path(".")
    else:
        output_dir = Path(output_dir)

    os.makedirs(output_dir / "chunks", exist_ok=True)

    stored_audio = defaultdict(bytes)
    starts = defaultdict(float)
    contains_activity = defaultdict(bool)
    total_time = defaultdict(float)

    if use_vad:
        vad = load_silero_vad()
    else:
        vad = None

    counter = 0

    while not done.value:
        data = builder_queue.get()
        id, start, stop, frames, sample_rate, sample_width, channels = (
            data.id,
            data.start,
            data.stop,
            data.frames,
            data.sample_rate,
            data.sample_width,
            data.channels
        )
        
        wf = wave.open(str(output_dir / "chunks" / "vad_tmp.wav"), 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(frames)
        wf.close()

        if use_vad:
            try:
                audio = read_audio(str(output_dir / "chunks" / "vad_tmp.wav"))
                activity = len(get_speech_timestamps(audio, vad)) > 0
            except RuntimeError:
                activity = False
        else:
            activity = True

        if not activity and not contains_activity[id]:
            starts[id] = start
            stored_audio[id] = frames
            total_time[id] = stop - start
        else:
            if len(stored_audio[id]) == 0:
                starts[id] = start
            stored_audio[id] += frames
            total_time[id] += stop - start

        if activity:
            contains_activity[id] = True

        # if there is no activity but there was previous activity, make utterance
        if (not activity and contains_activity[id]) or total_time[id] > max_utterance_time:
            next_file = str(output_dir / "chunks" / f"{counter:08}.wav")
            wf = wave.open(next_file, 'wb')
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(stored_audio[id])
            wf.close()

            processor_queue.put(AsrUtteranceData(id, starts[id], stop, next_file))

            stored_audio[id] = b''
            contains_activity[id] = False
            total_time[id] = 0
            counter += 1


def process_utterances(queue: "mp.Queue[AsrUtteranceData]", done, print_output=False, output_queue=None):
    # model = faster_whisper.WhisperModel("large-v2", compute_type="float16")
    model = faster_whisper.WhisperModel("small", compute_type="float16")

    while not done.value:
        data = queue.get()
        name, start, stop, chunk_file = data.id, data.start, data.stop, data.audio_file
        
        segments, info = model.transcribe(chunk_file, language="en")
        transcription = " ".join(segment.text for segment in segments if segment.no_speech_prob < 0.5)  # Join segments into a single string

        if print_output:
            print(f'{name}: {transcription}')

        if output_queue is not None:
            output_queue.put((name, start, stop, transcription, chunk_file))


class AsrFeature(IFeature):
    def __init__(self, devices: list[BaseDevice], n_processors=1, csv_log_file=None):
        """
        devices should be of the form [(name, index), ...]
        """
        self.device_lookup = {d.get_id():d for d in devices}
        self.asr_output_queue = mp.Queue()

        utterance_builder_queue = mp.Queue()
        utterance_processor_queue = mp.Queue()
        self.done = mp.Value(c_bool, False)
        recorders = [d.create_recorder_process(utterance_builder_queue, self.done) for d in devices]
        builder = mp.Process(target = build_utterances, args=(utterance_builder_queue, utterance_processor_queue, self.done))
        processors = [mp.Process(target=process_utterances, args=(utterance_processor_queue, self.done), kwargs={"output_queue":self.asr_output_queue}) for _ in range(n_processors)]


        for i in recorders + processors + [builder]:
            i.start()
        
        self.logger = Logger(file=csv_log_file)
        self.logger.write_csv_headers("frame", "name", "start", "stop", "text", "audio_file")


    def processFrame(self, frame, frame_count, includeText):
        utterances = []

        for id, device in self.device_lookup.items():
            device.handle_frame(frame_count)

        while not self.asr_output_queue.empty():
            id, start, stop, text, audio_file = self.asr_output_queue.get()
            if len(text.strip()) > 0:
                utterances.append((id, start, stop, text, audio_file))
                self.logger.append_csv(frame_count, id, start, stop, text, audio_file)

        if includeText:
            cv2.putText(frame, "ASR is live", (50,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        return utterances
