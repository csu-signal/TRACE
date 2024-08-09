@dataclass
class UtteranceInfo:
    utterance_id: int
    frame_received: int
    speaker_id: str
    text: str
    start_frame: int
    stop_frame: int
    audio_file: str | Path
