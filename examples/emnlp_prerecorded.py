import os
from pathlib import Path

from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

from mmdemo.demo import Demo
from mmdemo.features import (
    AccumulatedSelectedObjects,
    CommonGroundTracking,
    DenseParaphrasedTranscription,
    DisplayFrame,
    EMNLPFrame,
    GazeBodyTracking,
    Gesture,
    Log,
    Move,
    Object,
    Proposition,
    RecordedAudio,
    SaveVideo,
    SelectedObjects,
    VADUtteranceBuilder,
    WhisperTranscription,
)

MKV_PATH = Path(
    rf"F:\Weights_Task\Data\Fib_weights_original_videos\Group_01-master.mkv"
)
MKV_END = 5 * 60 + 30
AUDIO_PATH = Path(rf"F:\Weights_Task\Data\Group_01-audio.wav")
PLAYBACK_FRAME_RATE = 5

OUTPUT_FRAMES = Path("output_frames.mp4")
FINAL_OUTPUT = Path("final_output.mp4")


def convert_audio(input_path, output_path):
    """
    Ensure the input audio will be in the correct format using ffmpeg.
    Also normalizes the loudness of the audio.
    """
    os.system(
        f"ffmpeg -i {input_path} -filter:a loudnorm -ar 16000 -ac 1 -acodec pcm_s16le {output_path}"
    )


def add_audio_to_video(video, audio, output):
    """
    Add audio to a video using ffmpeg
    """
    os.system(
        f"ffmpeg -i {video} -i {audio} -map 0:v -map 1:a -c:v copy -shortest {output}"
    )


if __name__ == "__main__":
    color, depth, body_tracking, calibration = create_azure_kinect_features(
        DeviceType.PLAYBACK,
        mkv_path=MKV_PATH,
        playback_end_seconds=MKV_END,
        playback_frame_rate=PLAYBACK_FRAME_RATE,
    )

    converted_audio_file = Path("audio_wtd_converted.wav")
    convert_audio(AUDIO_PATH, converted_audio_file)

    audio = RecordedAudio(color, path=converted_audio_file)
    vad_utt_audio = VADUtteranceBuilder(audio, delete_input_files=True)
    transcriptions = WhisperTranscription(vad_utt_audio)

    gaze = GazeBodyTracking(body_tracking, calibration)
    gesture = Gesture(color, depth, body_tracking, calibration)

    obj = Object(color, depth, calibration)
    sel_obj = SelectedObjects(obj, gesture)

    ref_obj = AccumulatedSelectedObjects(sel_obj, transcriptions)

    dense = DenseParaphrasedTranscription(transcriptions, ref_obj)

    prop = Proposition(dense)
    move = Move(dense, vad_utt_audio)

    cgt = CommonGroundTracking(move, prop)

    output = EMNLPFrame(color, gaze, gesture, sel_obj, cgt, calibration)

    demo = Demo(
        targets=[
            DisplayFrame(output),
            SaveVideo(output, frame_rate=PLAYBACK_FRAME_RATE, video_name=OUTPUT_FRAMES),
            Log(dense, prop, move, csv=True),
            Log(transcriptions, stdout=True),
        ]
    )

    demo.show_dependency_graph()
    demo.run()
    add_audio_to_video(OUTPUT_FRAMES, converted_audio_file, FINAL_OUTPUT)

    demo.print_time_benchmarks()
