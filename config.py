# dll files needed to run azure kinect
K4A_DIR = r"C:\Program Files\Azure Kinect Body Tracking SDK\tools"

# output fps for processed recordings
PLAYBACK_TARGET_FPS = 5
PLAYBACK_SKIP_FRAMES = 30 // PLAYBACK_TARGET_FPS - 1

# data/model information for Weights Task Dataset evaluation
WTD_MKV_PATH = r"F:\Weights_Task\Data\Fib_weights_original_videos\Group_{0:02}-master.mkv"
WTD_AUDIO_PATH = r"F:\Weights_Task\Data\Group_{0:02}-audio.wav"
WTD_PROP_MODEL_PATH = r"F:\brady_wtd_eval_models\prop{0:02}"
WTD_MOVE_MODEL_PATH = r"F:\brady_wtd_eval_models\move{0:02}"
