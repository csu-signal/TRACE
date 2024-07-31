import argparse
from pathlib import Path
from demo.featureModules.utils import loadUtteranceFeatures
import os
import csv
import re
import wave

def initalizeCsv(path):
    if os.path.exists(path):
        os.remove(path)

    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["utterance_id", "frame_received", "speaker_id", "text", "start_frame", "stop_frame", "audio_file"])
    return 

def LogCsv(path, utterance_id, frame_received, speaker_id, text, start_frame, stop_frame, audio_file):
    with open(path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([utterance_id, frame_received, speaker_id, text, start_frame, stop_frame, audio_file])
    return 


def create_utterance_input(utterancePath, audio_file, outputFile, output_chunk_dir):
    output_chunk_dir = Path(output_chunk_dir)
    os.makedirs(output_chunk_dir, exist_ok=True)

    tmp_audio_file = output_chunk_dir / "full_recording.wav"
    os.system(f"ffmpeg -i {audio_file} -filter:a loudnorm -ar 16000 -ac 1 -acodec pcm_s16le {tmp_audio_file}")

    initalizeCsv(outputFile)

    utteranceFeatures = loadUtteranceFeatures(utterancePath)
    assert utteranceFeatures is not None

    subs = [
            (r"\bten\b", "10"),
            (r"\btwenty\b", "20"),
            (r"\bthirty\b", "30"),
            (r"\bforty\b", "40"),
            (r"\bfifty\b", "50")
        ]

    #for each ground truth utterance log the ASR values
    count = 1
    for u in utteranceFeatures:
        # print(u)
        startTime = float(u[1])
        endTime = float(u[2])
        startFrame = int(startTime * 30)
        endFrame = int(endTime * 30)

        text = u[3]
        for i,j in subs:
            text = re.sub(i, j, text, flags=re.IGNORECASE)


        chunk_path = str(output_chunk_dir / f"chunk{count:04}.wav")
        with wave.open(str(tmp_audio_file), "rb") as wf:
            with wave.open(chunk_path, "wb") as wf2:
                wf.readframes(int(wf.getframerate() * startTime))
                chunk = wf.readframes(int(wf.getframerate() * (endTime - startTime)))
                
                wf2.setparams(wf.getparams())
                wf2.writeframes(chunk)

        LogCsv(outputFile, count, endFrame, "Group", text, startFrame, endFrame, chunk_path)
        count += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--utterancePath', nargs='?', default="F:\\Weights_Task\\Data\\GAMR\\Utterances\\Group_01.csv")
    parser.add_argument('--audioFile', nargs='?', default="F:\\Weights_Task\\Data\\Group_01-audio.wav")
    parser.add_argument('--outputFile', nargs='?', default="F:\\Weights_Task\\Data\\FactPostProcessing\\Utterances\\Group_01.csv")
    parser.add_argument('--outputChunkDir', nargs='?', default="F:\\Weights_Task\\Data\\FactPostProcessing\\Utterances\\Group_01_chunks")
    args = parser.parse_args()

    create_utterance_input(args.utterancePath, args.audioFile, args.outputFile, args.outputChunkDir)
