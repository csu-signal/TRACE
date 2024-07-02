import pyaudio
import wave
import faster_whisper
import os
from colorama import Fore, Style, init
import time

from multiprocessing import Process, Queue, Value
from ctypes import c_bool

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


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

def record_chunks(device_name, device_index, queue, done, chunk_length=5, rate=16000, chunk=1024, format=pyaudio.paInt16):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024, input_device_index=device_index)  # Use the selected device index
    counter = 0
    while not done.value:
        next_file = fr"chunks\device{device_index}-{counter:05}.wav"
        counter += 1
        frames = []
        start = time.time()
        for i in range(0, int(rate / chunk * chunk_length)):
            data = stream.read(chunk)
            frames.append(data)
        stop = time.time()
        wf = wave.open(next_file, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        queue.put((device_name, start, stop, next_file))

    stream.stop_stream()
    stream.close()
    p.terminate()

def process_chunks(queue, done, print_output=False, output_queue=None):
    model = faster_whisper.WhisperModel("large-v2", compute_type="float16")
    while not done.value:
        name, start, stop, chunk_file = queue.get()

        segments, info = model.transcribe(chunk_file, language="en")
        transcription = " ".join(segment.text for segment in segments if segment.no_speech_prob < 0.5)  # Join segments into a single string

        if print_output:
            print(f'{name}: {transcription}')

        if output_queue is not None:
            output_queue.put((name, start, stop, transcription))

        os.remove(chunk_file)


if __name__ == "__main__":
    # Initialize colorama for colored text output
    init(autoreset=True)

    device_index = select_audio_device() # Get the selected device index
        
    queue = Queue()
    done = Value(c_bool, False)

    recorder = Process(target=record_chunks, args=("recording", device_index, queue, done))
    processor = Process(target=process_chunks, args=(queue, done), kwargs={"print_output":True})

    recorder.start()
    processor.start()

    recorder.join()
    processor.join()
