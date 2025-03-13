from kokoro import KPipeline
import soundfile as sf
import sounddevice as sd
import torch
import os
import random

pipeline = KPipeline(lang_code='a') # <= make sure lang_code matches voice
# voice_tensor = torch.load('am_michael.pt', weights_only=True)

def generate_speech(text,save=False):
    generator = pipeline(
        text, voice='am_michael', # <= change voice here
        speed=1, split_pattern=r'\n+'
    )

    # This might be faster
    # Alternatively, load voice tensor directly:
    # voice_tensor = torch.load('am_michael.pt', weights_only=True)
    # generator = pipeline(
    #     text, voice=voice_tensor,
    #     speed=1, split_pattern=r'\n+'
    # )

    for i, (gs, ps, audio) in enumerate(generator):
        # print(i)  # i => index
        # print(gs) # gs => graphemes/text
        # print(ps) # ps => phonemes
        if(save):
            sf.write(f'audio/{text}.wav', audio, 24000) # save each audio file
        sd.play(audio, 24000)
        sd.wait()


text = ''
while(text != "Stop"):
    text = input("Please enter your text: ")
    opening = random.choice(os.listdir("audio"))
    audio, samplerate = sf.read(fr"audio\{opening}")
    sd.play(audio, 24000)
    sd.wait()
    generate_speech(text)