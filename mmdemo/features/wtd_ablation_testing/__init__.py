"""
Features for ablation testing of the Weights Task Dataset. These features
output ground truth data from the dataset. The required csv inputs to these
features can be generated using the script at
"scripts/wtd_annotations/create_all_wtd_inputs.py".
"""

from mmdemo.features.wtd_ablation_testing.gesture_feature import (
    GestureSelectedObjectsGroundTruth,
)
from mmdemo.features.wtd_ablation_testing.object_feature import ObjectGroundTruth
from mmdemo.features.wtd_ablation_testing.transcription_feature import (
    create_transcription_and_audio_ground_truth_features,
)
