from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (  # ASRInterface,; BodyTrackingInterface,; ColorImageInterface,; DenseParaphraseInterface,; DepthImageInterface,; GestureInterface,; ObjectInterface,; UtteranceChunkInterface,
    SelectedObjectsInterface,
    TranscriptionInterface,
)

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class DenseParaphrasing(BaseFeature[TranscriptionInterface]):
    @classmethod
    def get_input_interfaces(cls):
        return [
            SelectedObjectsInterface,
            TranscriptionInterface,
        ]

    @classmethod
    def get_output_interface(cls):
        return TranscriptionInterface

    def initialize(self):
        # self.paraphrased_utterance_lookup: dict[int, UtteranceInfo] = {}

        # if log_dir is not None:
        #     self.logger = Logger(file=log_dir / self.LOG_FILE)
        # else:
        #     self.logger = Logger()
        # self.logger.write_csv_headers("frame", "utterance_id", "updated_text", "old_text", "subs_made")
        pass

    def get_output(
        self,
        select_obj: SelectedObjectsInterface,
        tran: TranscriptionInterface,
    ):
        if not select_obj.is_new() and not tran.is_new():
            return None

        # call prop extractor, create interface, and return

    # def processFrame(self, frame, new_utterances: list[int], utterance_lookup: dict[int, UtteranceInfo], blockCache, frame_count):
    #     clear = False
    #     for i in new_utterances:
    #         utterance_info = utterance_lookup[i]
    #         text = utterance_info.text

    #         plural_demo_regex = r"\b(" + "|".join([d.text for d in demonstratives if d.plural]) + r")\b"
    #         singlular_demo_regex = r"\b(" + "|".join([d.text for d in demonstratives if not d.plural]) + r")\b"

    #         key = get_frame_bin(utterance_info.start_frame)
    #         stop_bin = get_frame_bin(utterance_info.stop_frame)
    #         while key <= stop_bin:
    #             if key in blockCache and len(blockCache[key]) > 0:
    #                 targets = blockCache[key]

    #                 text = re.sub(plural_demo_regex, ", ".join(targets), text, count=1, flags=re.IGNORECASE)

    #                 for i in range(len(targets)):
    #                     text = re.sub(singlular_demo_regex, targets[i], text, count=1, flags=re.IGNORECASE)

    #             key+=1

    #         self.paraphrased_utterance_lookup[utterance_info.utterance_id] = UtteranceInfo(
    #                 utterance_info.utterance_id,
    #                 frame_count,
    #                 utterance_info.speaker_id,
    #                 text,
    #                 utterance_info.start_frame,
    #                 utterance_info.stop_frame,
    #                 utterance_info.audio_file
    #             )

    #         self.logger.append_csv(
    #                 frame_count,
    #                 utterance_info.utterance_id,
    #                 text,
    #                 utterance_info.text,
    #                 text.lower() != utterance_info.text.lower()
    #             )
