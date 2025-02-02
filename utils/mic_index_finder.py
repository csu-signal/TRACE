# Stand alone file to display intput device index.
# The number associated with the input device needed should be passed to MicAudio in the emnlp_live file.

import pyaudio


def list_microphones():
    p = pyaudio.PyAudio()

    # Get the number of audio input devices
    device_count = p.get_device_count()

    microphones = []
    for i in range(device_count):
        device_info = p.get_device_info_by_index(i)

        # Check if the device is an input (microphone)
        if device_info["maxInputChannels"] > 0:
            microphones.append((i, device_info["name"]))

    p.terminate()

    return microphones


# List microphones and their indices
mic_list = list_microphones()
for index, name in mic_list:
    print(f"Index: {index}, Name: {name}")
