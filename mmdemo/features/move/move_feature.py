from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import DenseParaphraseInterface, MoveInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class Move(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return []

    @classmethod
    def get_output_interface(cls):
        return MoveInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(self, t: DenseParaphraseInterface):
        if not t.is_new():
            return None

        # call move classifier, create interface, and return

    # def log_move(self, frame_count, move: MoveInfo, output, text, audio_file):
    #     self.logger.append_csv(
    #             frame_count,
    #             move.utterance_id,
    #             int("STATEMENT" in move.move),
    #             int("ACCEPT" in move.move),
    #             int("DOUBT" in move.move),
    #             output,
    #             text,
    #             audio_file)

    # def update_bert_embeddings(self, text):
    #     input_ids = torch.tensor(self.tokenizer.encode(text), device=self.device).unsqueeze(0)
    #     cls_embeddings = self.bert_model(input_ids)[0][:, 0]

    #     self.bert_embedding_history = torch.cat([self.bert_embedding_history[1:], cls_embeddings])

    # def update_smile_embeddings(self, audio_file):
    #     if os.path.exists(audio_file):
    #         embedding = torch.tensor(self.smile.process_file(audio_file).to_numpy(), device=self.device)
    #     else:
    #         embedding = torch.zeros(1, SMILE_EMBEDDING_DIM, device=self.device)

    #     self.opensmile_embedding_history = torch.cat([self.opensmile_embedding_history[1:], embedding])

    # def processFrame(self, frame, new_utterances: list[int], utterance_lookup: list[UtteranceInfo] | dict[int, UtteranceInfo], frameIndex, includeText):
    #     for i in new_utterances:
    #         text = utterance_lookup[i].text
    #         audio_file = utterance_lookup[i].audio_file

    #         self.update_bert_embeddings(text)
    #         in_bert = self.bert_embedding_history

    #         self.update_smile_embeddings(audio_file)
    #         in_open = self.opensmile_embedding_history

    #         # TODO: other inputs for move classifier
    #         in_cps = torch.zeros((UTTERANCE_HISTORY_LEN, 3), device=self.device)
    #         in_action = torch.zeros((UTTERANCE_HISTORY_LEN, 78), device=self.device)
    #         in_gamr = torch.zeros((UTTERANCE_HISTORY_LEN, 243), device=self.device)

    #         out = torch.sigmoid(self.model(in_bert, in_open, in_cps, in_action, in_gamr, hyperparam, modalities))
    #         out = out.cpu().detach().numpy()

    #         present_class_indices = (out > 0.5)
    #         move = [self.class_names[idx] for idx, class_present in enumerate(present_class_indices) if class_present]
    #         self.move_lookup[i] = MoveInfo(i, move)

    #         self.log_move(frameIndex, self.move_lookup[i], out, text, audio_file)

    #     if includeText:
    #         cv2.putText(frame, "Move classifier is live", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
