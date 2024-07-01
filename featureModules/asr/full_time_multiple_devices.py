import multiprocessing as mp
from full_time_recording import main
import pyaudio
from ctypes import c_bool
import time

def start_transcription_process(device_index, num_devices, done):
    main(device_index=device_index, done=done)

def get_device_indices(devices=None):
    print("Available devices:")
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        print(i, ":", p.get_device_info_by_index(i).get('name'))
    devices = input("Enter device indices separated by comma (e.g., 0,1,2): ")
    return [int(index.strip()) for index in devices.split(',')]

if __name__ == "__main__":
    device_indices = get_device_indices()
    num_devices = len(device_indices)

    done = mp.Value(c_bool, False)

    processes = [mp.Process(target=start_transcription_process, args=(device_index, num_devices, done)) for device_index in device_indices]

    for process in processes:
        process.start()

    time.sleep(100)
    done.value = True
    print("closing processes")

    for process in processes:
        process.join()
