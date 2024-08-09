UTTERANCE_HISTORY_LEN = 4

BERT_EMBEDDING_DIM = 768
SMILE_EMBEDDING_DIM = 88


@dataclass
class MoveInfo:
    utterance_id: int
    move: list[str]
