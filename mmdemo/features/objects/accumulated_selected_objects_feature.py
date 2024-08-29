import time
from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import SelectedObjectsInterface, TranscriptionInterface
from mmdemo.utils.frame_time_converter import FrameTimeConverter


# TODO: this feature is not the most efficient because it's memory can grow very large.
# Some sort of binning or clearing old memory could be implemented, but this is very
# prone to errors so write good tests if this is done.
@final
class AccumulatedSelectedObjects(BaseFeature[SelectedObjectsInterface]):
    """
    Determine which objects are being referenced in a given transcription
    by accumulating the selected objects over the time of the transcription.

    Input interfaces are `SelectedObjectsInterface` and `TranscriptionInterface`.

    Output interface is `SelectedObjectsInterface`.
    """

    def __init__(
        self,
        selected_objects: BaseFeature[SelectedObjectsInterface],
        transcription: BaseFeature[TranscriptionInterface],
    ):
        super().__init__(selected_objects, transcription)

    def initialize(self):
        self.internal_frame_count = 0
        self.frame_time_converter = FrameTimeConverter()
        self.saved_object_data: dict[int, SelectedObjectsInterface] = {}

    def get_output(
        self,
        selected_objects: SelectedObjectsInterface,
        transcription: TranscriptionInterface,
    ):
        if selected_objects.is_new():
            self.frame_time_converter.add_data(self.internal_frame_count, time.time())
            self.saved_object_data[self.internal_frame_count] = selected_objects
            self.internal_frame_count += 1

        if not transcription.is_new():
            return None

        # if there are no objects seen yet, return an empty interface
        if self.frame_time_converter.get_num_datapoints() == 0:
            return SelectedObjectsInterface(objects=[])

        objects_seen = set()
        objects = []

        start_frame = self.frame_time_converter.get_frame(transcription.start_time)
        end_frame = self.frame_time_converter.get_frame(transcription.end_time)
        for i in range(start_frame, end_frame + 1):
            if i not in self.saved_object_data:
                continue

            selected = self.saved_object_data[i]
            for info, sel in selected.objects:
                if not sel:
                    continue

                if info.object_class not in objects_seen:
                    objects.append((info, True))
                    objects_seen.add(info.object_class)

        return SelectedObjectsInterface(objects=objects)
