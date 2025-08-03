"""
Premade features
"""

from mmdemo.features.common_ground.cgt_feature import CommonGroundTracking
from mmdemo.features.common_ground.dpip_cgt_output import DpipCommonGroundTracking
from mmdemo.features.common_ground.action_feature import DpipActionFeature
from mmdemo.features.gaze.gaze_body_tracking_feature import GazeBodyTracking
from mmdemo.features.gaze.aaai_gaze_body_tracking_feature import AaaiGazeBodyTracking
from mmdemo.features.gesture.gesture_feature import Gesture
from mmdemo.features.gesture.aaai_gesture_feature import AaaiGesture
from mmdemo.features.move.move_feature import Move
from mmdemo.features.objects.accumulated_selected_objects_feature import (
    AccumulatedSelectedObjects,
)
from mmdemo.features.objects.object_feature import Object
from mmdemo.features.objects.dpip_object_feature import DpipObject
from mmdemo.features.objects.selected_objects_feature import SelectedObjects

from mmdemo.features.outputs.display_frame_feature import DisplayFrame
from mmdemo.features.outputs.emnlp_frame_feature import EMNLPFrame
from mmdemo.features.outputs.dpip_frame_feature import DpipFrame
from mmdemo.features.outputs.hcii_it_frame_feature import HCII_IT_Frame
from mmdemo.features.outputs.logging_feature import Log
from mmdemo.features.outputs.save_video_feature import SaveVideo
from mmdemo.features.proposition.prop_feature import Proposition
from mmdemo.features.proposition.dpip_prop_feature import DpipProposition
from mmdemo.features.transcription.dense_paraphrasing_feature import (
    DenseParaphrasedTranscription,
)
from mmdemo.features.transcription.whisper_transcription_feature import (
    WhisperTranscription,
)
from mmdemo.features.utterance.audio_input_features import MicAudio, RecordedAudio
from mmdemo.features.utterance.vad_builder_feature import VADUtteranceBuilder
from mmdemo.features.planner.planner_feature import Planner

#add feature pose created by CSU
from mmdemo.features.pose.pose_feature import Pose
from mmdemo.features.pose.selected_participant_feature import SelectedParticipant

#new features created by Zilong
from mmdemo.features.gaze_event.gaze_event_decision_feature import GazeEvent
from mmdemo.features.gaze_selection.gaze_selection_feature import GazeSelection
from mmdemo.features.pose_event.pose_event_decision_feature import PoseEvent
from mmdemo.features.gesture_event.gesture_event_decision_feature import GestureEvent
from mmdemo.features.engagement_level.engagement_level_feature import EngagementLevel
from mmdemo.features.outputs.aaai_frame_feature import AAAIFrame

# TODO work with Mariah to get them added and update the yaml
from mmdemo.features.speech_output.speechoutput_feature import SpeechOutput
from mmdemo.features.speech_output.dpipSpeechoutput_feature import DpipSpeechOutput
from mmdemo.features.outputs.user_study_frame_feature import UserFrame
