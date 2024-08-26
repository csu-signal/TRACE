"""
Premade features
"""

from mmdemo.features.utterance.audio_input_features import MicAudio, RecordedAudio
from mmdemo.features.utterance.vad_builder_feature import VADUtteranceBuilder
from mmdemo.features.transcription.whisper_transcription_feature import (
    WhisperTranscription,
)
from mmdemo.features.proposition.prop_feature import Proposition
from mmdemo.features.move.move_feature import Move
from mmdemo.features.common_ground.cgt_feature import CommonGroundTracking

from mmdemo.features.dense_paraphrasing.referenced_objects_feature import (
    ReferencedObjects,
)
from mmdemo.features.dense_paraphrasing.dense_paraphrasing_feature import (
    DenseParaphrasing,
)

from mmdemo.features.gaze.gaze_body_tracking_feature import GazeBodyTracking
from mmdemo.features.gesture.gesture_feature import Gesture
from mmdemo.features.objects.object_feature import Object
from mmdemo.features.objects.selected_objects_feature import SelectedObjects

from mmdemo.features.outputs.emnlp_frame_feature import EMNLPFrame
from mmdemo.features.outputs.display_frame_feature import DisplayFrame
from mmdemo.features.outputs.logging_feature import Log
