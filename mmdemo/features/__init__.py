"""
Premade features
"""

from mmdemo.features.common_ground.cgt_feature import CommonGroundTracking
from mmdemo.features.gaze.gaze_body_tracking_feature import GazeBodyTracking
from mmdemo.features.gesture.gesture_feature import Gesture
from mmdemo.features.move.move_feature import Move
from mmdemo.features.objects.accumulated_selected_objects_feature import (
    AccumulatedSelectedObjects,
)
from mmdemo.features.objects.object_feature import Object
from mmdemo.features.objects.selected_objects_feature import SelectedObjects

from mmdemo.features.outputs.display_frame_feature import DisplayFrame
from mmdemo.features.outputs.emnlp_frame_feature import EMNLPFrame
from mmdemo.features.outputs.hcii_it_frame_feature import HCII_IT_Frame
from mmdemo.features.outputs.logging_feature import Log
from mmdemo.features.outputs.save_video_feature import SaveVideo
from mmdemo.features.proposition.prop_feature import Proposition
from mmdemo.features.transcription.dense_paraphrasing_feature import (
    DenseParaphrasedTranscription,
)
from mmdemo.features.transcription.whisper_transcription_feature import (
    WhisperTranscription,
)
from mmdemo.features.utterance.audio_input_features import MicAudio, RecordedAudio
from mmdemo.features.utterance.vad_builder_feature import VADUtteranceBuilder

#TODO figure out why I'm missing planner packages?
#from mmdemo.features.planner.planner_feature import Planner