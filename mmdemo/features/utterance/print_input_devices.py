import pyaudio

if __name__ == "__main__":
    p = pyaudio.PyAudio()
    print("Available devices:")
    for i in range(p.get_device_count()):
        print(i, ":", p.get_device_info_by_index(i).get("name"))
