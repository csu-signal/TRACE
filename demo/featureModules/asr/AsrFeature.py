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

from demo.featureModules.asr.device import AsrDeviceData, BaseDevice
from demo.featureModules.IFeature import *
from demo.logger import Logger
from demo.featureModules.utils import *

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

@dataclass
class AsrUtteranceData:
    id: str
    start_time: float
    stop_time: float
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

    while True:
        data = builder_queue.get()
        if data is None:
            break
        id, start, stop, frames, sample_rate, sample_width, channels = (
            data.id,
            data.start_time,
            data.stop_time,
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

    processor_queue.put(None)


def process_utterances(queue: "mp.Queue[AsrUtteranceData]", print_output=False, output_queue=None):
    # model = faster_whisper.WhisperModel("large-v2", compute_type="float16")
    model = faster_whisper.WhisperModel("small", compute_type="float16")

    while True:
        data = queue.get()
        if data is None:
            break
        name, start_time, stop_time, chunk_file = data.id, data.start_time, data.stop_time, data.audio_file
        
        segments, info = model.transcribe(chunk_file, language="en")
        transcription = " ".join(segment.text for segment in segments if segment.no_speech_prob < 0.5)  # Join segments into a single string

        if print_output:
            print(f'{name}: {transcription}')

        if output_queue is not None:
            output_queue.put((name, start_time, stop_time, transcription, chunk_file))

    if output_queue is not None:
        output_queue.put(None)


@dataclass
class UtteranceInfo:
    utterance_id: int
    frame_received: int
    speaker_id: str
    text: str
    start_frame: int
    stop_frame: int
    audio_file: str | Path

class AsrFeature(IFeature):
    LOG_FILE = "asrOutput.csv"
    
    def __init__(self, devices: list[BaseDevice], n_processors=1, log_dir=None):
        """
        devices should be of the form [(name, index), ...]
        """
        self.device_lookup = {d.get_id():d for d in devices}
        self.asr_output_queue = mp.Queue()
        
        self.utterance_builder_queue = mp.Queue()
        self.utterance_processor_queue = mp.Queue()
        self.done = mp.Value(c_bool, False)
        self.recorders = [d.create_recorder_process(self.utterance_builder_queue, self.done) for d in devices]
        self.builder = mp.Process(target = build_utterances, args=(self.utterance_builder_queue, self.utterance_processor_queue ), kwargs={"output_dir": log_dir})

        self.n_processors = n_processors
        self.processors = [mp.Process(target=process_utterances, args=(self.utterance_processor_queue,), kwargs={"output_queue":self.asr_output_queue}) for _ in range(self.n_processors)]


        for i in self.recorders + self.processors + [self.builder]:
            i.start()

        self.init_logger(log_dir)

        self.utterance_lookup: dict[int, UtteranceInfo] = {}

    def exit(self, join_processes=True):
        self.done.value = True

        while self.asr_output_queue.get() is not None:
            pass

        if join_processes:
            for i in self.recorders + self.processors + [self.builder]:
                i.join()


    def init_logger(self, log_dir):
        if log_dir is not None:
            self.logger = Logger(file=log_dir / self.LOG_FILE)
        else:
            self.logger = Logger()

        self.logger.write_csv_headers("utterance_id", "frame_received", "speaker_id", "text", "start_frame", "stop_frame", "audio_file")

    def log_utterance(self, ut: UtteranceInfo):
        self.logger.append_csv(ut.utterance_id, ut.frame_received, ut.speaker_id, ut.text, ut.start_frame, ut.stop_frame, ut.audio_file)

    def processFrame(self, frame, frame_count, time_to_frame, includeText):
        new_utterance_ids = []

        for speaker, device in self.device_lookup.items():
            device.handle_frame(frame_count)

        while not self.asr_output_queue.empty():
            speaker, start_time, stop_time, text, audio_file = self.asr_output_queue.get()
            if len(text.strip()) > 0:
                utterance = UtteranceInfo(
                        len(self.utterance_lookup),
                        frame_count,
                        speaker,
                        text,
                        time_to_frame(start_time),
                        time_to_frame(stop_time),
                        audio_file
                    )
                self.utterance_lookup[utterance.utterance_id] = utterance
                
                self.log_utterance(utterance)

                new_utterance_ids.append(utterance.utterance_id)

        if includeText:
            cv2.putText(frame, "ASR is live", (50,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

        return new_utterance_ids