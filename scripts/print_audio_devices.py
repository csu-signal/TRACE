import pyaudio

if __name__ == "__main__":
    p = pyaudio.PyAudio()
    print("Available input devices:")
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info["maxInputChannels"] > 0:
            print(i, ":", p.get_device_info_by_index(i).get("name"))
