import pyaudio
import wave
import whisperx
import os
from colorama import Fore, Style, init

# Initialize colorama for colored text output
init(autoreset=True)


def record_chunk(p, stream, file_path, device_index, chunk_length=3, rate=16000, chunk=1024, format=pyaudio.paInt16):
    frames = []
    for i in range(0, int(rate / chunk * chunk_length)):
        data = stream.read(chunk)
        frames.append(data)
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_chunk(model, chunk_file):
    audio = whisperx.load_audio(chunk_file)
    result = model.transcribe(audio)
    return result["segments"]

def main(device_index=None, identifier="", num_devices=1):
    if device_index is None:
        device_index = select_audio_device() # Get the selected device index
        
    model = whisperx.load_model("large-v2", device="cuda", compute_type="float16", language="en")
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024, input_device_index=device_index)  # Use the selected device index
    accumulated_transcript = ""

    try: 
        while True:
            # Generate a unique filename for the temporary chunk
            chunk_file = f'temp_chunk_{device_index}.wav'
            record_chunk(p, stream, chunk_file, device_index)  # Pass the device index to record_chunk
            segments = transcribe_chunk(model, chunk_file)  # Transcribe the chunk
            transcription = " ".join(segment['text'] for segment in segments)  # Join segments into a single string
            colors = [Fore.RED, Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]
            color_index = device_index % len(colors) 

            print(f'{colors[color_index]}{device_index}: {transcription}')
            os.remove(chunk_file)

            accumulated_transcript += transcription + " "

    except KeyboardInterrupt:
        print("KeyboardInterrupt")

        with open(f"logs/log_{identifier}.txt", "w") as f:
            f.write(accumulated_transcript)
    finally:
        print(f"{identifier}:"+ accumulated_transcript)
        #print("log:" + accumulated_transcript)
        stream.stop_stream()
        stream.close()
        p.terminate()

    # Ensure the unique temporary chunk file is removed after processing
    if os.path.exists(chunk_file):
        os.remove(chunk_file)

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

if __name__ == "__main__":
    main()