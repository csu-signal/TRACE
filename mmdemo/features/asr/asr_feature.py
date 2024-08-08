from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import TranscriptionInterface, UtteranceChunkInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class ASR(BaseFeature):
    def __init__(self, *args):
        super().__init__()
        self.register_dependencies(
            [TranscriptionInterface], args
        )  # TODO Check if this is needed

    @classmethod
    def get_output_interface(cls):
        return UtteranceChunkInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: TranscriptionInterface):
        if not t.is_new():
            return None

        # call __, create interface, and return

    # def init_logger(self, log_dir):
    #     if log_dir is not None:
    #         self.logger = Logger(file=log_dir / self.LOG_FILE)
    #     else:
    #         self.logger = Logger()

    #     self.logger.write_csv_headers("utterance_id", "frame_received", "speaker_id", "text", "start_frame", "stop_frame", "audio_file")

    # def log_utterance(self, ut: UtteranceInfo):
    #     self.logger.append_csv(ut.utterance_id, ut.frame_received, ut.speaker_id, ut.text, ut.start_frame, ut.stop_frame, ut.audio_file)

    # def processFrame(self, frame, frame_count, time_to_frame, includeText):
    #     new_utterance_ids = []

    #     for speaker, device in self.device_lookup.items():
    #         device.handle_frame(frame_count)

    #     while not self.asr_output_queue.empty():
    #         speaker, start_time, stop_time, text, audio_file = self.asr_output_queue.get()
    #         if len(text.strip()) > 0:
    #             utterance = UtteranceInfo(
    #                     len(self.utterance_lookup),
    #                     frame_count,
    #                     speaker,
    #                     text,
    #                     time_to_frame(start_time),
    #                     time_to_frame(stop_time),
    #                     audio_file
    #                 )
    #             self.utterance_lookup[utterance.utterance_id] = utterance

    #             self.log_utterance(utterance)

    #             new_utterance_ids.append(utterance.utterance_id)

    #     if includeText:
    #         cv2.putText(frame, "ASR is live", (50,350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

    #     return new_utterance_ids
