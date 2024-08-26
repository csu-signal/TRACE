import re
from dataclasses import dataclass
from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import SelectedObjectsInterface, TranscriptionInterface


@dataclass
class Demonstrative:
    text: str
    plural: bool


@final
class DenseParaphrasing(BaseFeature[TranscriptionInterface]):
    """
    Substitute demonstratives like "this" and "that" in a transcription with
    selected object names

    Input interface are `TranscriptionInterface`, `SelectedObjectsInterface`

    Output interface is `TranscriptionInterface`
    """

    def __init__(
        self,
        transcription: BaseFeature[TranscriptionInterface],
        selected_objects: BaseFeature[SelectedObjectsInterface],
    ):
        super().__init__(transcription, selected_objects)

    def initialize(self):
        demonstratives = [
            Demonstrative("those", True),
            Demonstrative("these", True),
            Demonstrative("this", False),
            Demonstrative("that", False),
            Demonstrative("it", False),
        ]

        self.plural_demo_regex = (
            r"\b(" + "|".join([d.text for d in demonstratives if d.plural]) + r")\b"
        )
        self.singular_demo_regex = (
            r"\b(" + "|".join([d.text for d in demonstratives if not d.plural]) + r")\b"
        )

    def get_output(
        self,
        transcription: TranscriptionInterface,
        selected_objects: SelectedObjectsInterface,
    ):
        if not selected_objects.is_new() or not transcription.is_new():
            return None

        text = transcription.text

        targets = [
            obj.object_class.value for obj, sel in selected_objects.objects if sel
        ]

        if len(targets) > 0:
            text = re.sub(
                self.plural_demo_regex,
                ", ".join(targets),
                text,
                count=1,
                flags=re.IGNORECASE,
            )

        for i in range(len(targets)):
            text = re.sub(
                self.singular_demo_regex,
                targets[i],
                text,
                count=1,
                flags=re.IGNORECASE,
            )

        return TranscriptionInterface(
            speaker_id=transcription.speaker_id,
            start_time=transcription.start_time,
            end_time=transcription.end_time,
            text=text,
        )
