from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import TranscriptionInterface  # , UtteranceChunkInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class Transcription(BaseFeature[TranscriptionInterface]):
    # LOG_FILE = "asrOutput.csv"
    @classmethod
    def get_input_interfaces(cls):
        return []  # [UtteranceChunkInterface]

    @classmethod
    def get_output_interface(cls):
        return TranscriptionInterface

    def initialize(self):
        # self.device_lookup = {d.get_id():d for d in devices}
        # self.asr_output_queue = mp.Queue()

        # self.utterance_builder_queue = mp.Queue()
        # self.utterance_processor_queue = mp.Queue()
        # self.done = mp.Value(c_bool, False)
        # self.recorders = [d.create_recorder_process(self.utterance_builder_queue, self.done) for d in devices]
        # self.builder = mp.Process(target = build_utterances, args=(self.utterance_builder_queue, self.utterance_processor_queue ), kwargs={"output_dir": log_dir})

        # self.n_processors = n_processors
        # self.processors = [mp.Process(target=process_utterances, args=(self.utterance_processor_queue,), kwargs={"output_queue":self.asr_output_queue}) for _ in range(self.n_processors)]

        # for i in self.recorders + self.processors + [self.builder]:
        #     i.start()

        # self.init_logger(log_dir)

        # self.utterance_lookup: dict[int, UtteranceInfo] = {}
        pass

    def get_output(
        self,
        t: TranscriptionInterface,
    ):  # s: UtteranceChunkInterface):
        if not t.is_new():  # or not s.is_new():
            return None

        # call __, create interface, and return

    # def exit(self, join_processes=True):
    #     self.done.value = True

    #     while self.asr_output_queue.get() is not None:
    #         pass

    #     if join_processes:
    #         for i in self.recorders + self.processors + [self.builder]:
    #             i.join()

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
