# conda activate C:\ProgramData\anaconda3\envs\asrEnv

import sys
import queue
import threading
import sounddevice as sd
import espnet
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
from espnet_model_zoo.downloader import ModelDownloader
import argparse
import numpy as np
import wave

tag='D-Keqi/espnet_asr_train_asr_streaming_transformer_raw_en_bpe500_sp_valid.acc.ave'
print("in")
d=ModelDownloader()
print("downloader set")
speech2text = Speech2TextStreaming(
    **d.download_and_unpack(tag),
    token_type=None,
    bpemodel=None,
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=20,
    ctc_weight=0.5,
    lm_weight=0.0,
    penalty=0.0,
    nbest=1,
    device = "cpu",
    disable_repetition_detection=True,
    decoder_text_length_limit=0,
    encoded_feat_length_limit=0
)
print("speech instance created")
prev_lines = 0
def progress_output(text):
    global prev_lines
    lines=['']
    for i in text:
        if len(lines[-1]) > 100:
            lines.append('')
        lines[-1] += i
    for i,line in enumerate(lines):
        if i == prev_lines:
            sys.stderr.write('\n\r')
        else:
            sys.stderr.write('\r\033[B\033[K')
        sys.stderr.write(line)

    prev_lines = len(lines)
    sys.stderr.flush()

def recognize(wavfile):
    with wave.open(wavfile, 'rb') as wavfile:
        ch=wavfile.getnchannels()
        bits=wavfile.getsampwidth()
        rate=wavfile.getframerate()
        nframes=wavfile.getnframes()
        buf = wavfile.readframes(-1)
        data=np.frombuffer(buf, dtype='int16')
    speech = data.astype(np.float16)/32767.0 #32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
    sim_chunk_length = 640
    if sim_chunk_length > 0:
        for i in range(len(speech)//sim_chunk_length):
            results = speech2text(speech=speech[i*sim_chunk_length:(i+1)*sim_chunk_length], is_final=False)
            if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                progress_output(nbests[0])
            else:
                progress_output("")
            
        results = speech2text(speech[(i+1)*sim_chunk_length:len(speech)], is_final=True)
    else:
        results = speech2text(speech, is_final=True)
    nbests = [text for text, token, token_int, hyp in results]
    progress_output(nbests[0])

def rec_sound(data, frames, time, status):
    global q
    q.put(data.astype(np.float16)/32767.0)

def asr():
    global q
    global keepWorking
    first = True

    while keepWorking:
        try:
            data = q.get(block=True, timeout=1 if not first else None)  # wait infinitely long on first call
            first = False
        except queue.Empty:
            break
        results = speech2text(speech=data, is_final=False)
        if results is not None and len(results) > 0:
            nbests = [text for text, token, token_int, hyp in results]
            text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
            #print(nbests[0])
            progress_output(nbests[0])
        else:
            #pass
            progress_output("")

q = queue.Queue()
keepWorking = True
t = threading.Thread(target=asr)
t.start()
count = 0
print("start")
with sd.InputStream(samplerate=16000, channels=1, blocksize=640, callback=rec_sound, dtype=np.int16):
    sd.sleep(20000)
print("stop")
keepWorking = False
t.join()
