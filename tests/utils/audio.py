import wave

def get_length(wav_path):
    """
    Helper function to get length of wav file
    """
    wf = wave.open(str(wav_path), "rb")
    length = wf.getnframes() / wf.getframerate()
    wf.close()
    return length

