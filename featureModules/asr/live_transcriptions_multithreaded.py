import multiprocessing
from live_transcription import main
import pyaudio

def start_transcription_process(device_index, num_devices):
    # Pass device_index and num_devices to main function for unique identification
    main(device_index, identifier=f"Microphone {device_index}", num_devices=num_devices)

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

    processes = [multiprocessing.Process(target=start_transcription_process, args=(device_index, num_devices)) for device_index in device_indices]

    for process in processes:
        process.start()

    for process in processes:
        process.join()