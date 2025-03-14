"""
Base feature definition
"""
from kokoro import KPipeline
import soundfile as sf
import sounddevice as sd
import torch
import os
import random
from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (FrictionOutputInterface,SpeechOutputInterface)

# T = TypeVar("T", bound=FrictionOutputInterface)


class SpeechOutput(BaseFeature[SpeechOutputInterface]):
    """
    The base class all features in the demo must implement.

    ***The input feature is the friction statement.
    ***The output feature is the text-to-speech audio file.
    """

    def __init__(self, friction: BaseFeature[FrictionOutputInterface]):
        # self._deps = []
        # self._rev_deps = []
        # self._register_dependencies(args)
        super().__init__(friction)
        self.speechoutput = False

    def initialize(self):
        """
        Initialize feature. This is where all the time/memory
        heavy initialization should go. Put it here instead of
        __init__ to avoid wasting resources when extra features
        exist. This method is guaranteed to be called before
        the first `get_output` and run on the main thread.

        ***Initialize the pipeline and load the voice
        """
        self.pipeline = KPipeline(lang_code='a')
        self.voice_tensor = torch.load('C:/GitHub/TRACE/mmdemo/features/speech_output/am_michael.pt', weights_only=True)
        self.last_friction = ''

    # def _register_dependencies(self, deps: "list[BaseFeature] | tuple"):
    #     """
    #     Add other features as dependencies which are required
    #     to be evaluated before this feature.

    #     Arguments:
    #     deps -- a list of dependency features

    #     ***Nothing here
    #     """
    #     assert len(self._deps) == 0, "Dependencies have already been registered"
    #     for d in deps:
    #         self._deps.append(d)
    #         d._rev_deps.append(self)

    def get_output(self,frictionout : FrictionOutputInterface):
        """
        Return output of the feature. The return type must be the output
        interface to provide new data and `None` if there is no new data.
        It is very important that this function does not modify any of the
        input interfaces because they may be reused for other features.

        Arguments:
        args -- list of output interfaces from dependencies in the order
                they were registered. Calling `.is_new()` on any of these
                elements will return True if the argument has not been
                sent before. It is possible that the interface will not
                contain any data before the first new data is sent.

        ***Check for new friction
        ***Generate speech
        """
        friction = frictionout.friction_statement
        friction = friction.split("Friction:")[-1].split("r*")[0]
        if friction == '' or friction == self.last_friction:
            return SpeechOutputInterface(speech_output=self.speechoutput)
        #play a request for interruption
        opening = random.choice(os.listdir("C:/GitHub/TRACE/mmdemo/features/speech_output/audio"))
        audio, samplerate = sf.read(fr"C:/GitHub/TRACE/mmdemo/features/speech_output/audio/{opening}")
        sd.play(audio, samplerate)
        sd.wait()
        #generate speech, splits on newline
        generator = self.pipeline(
        friction, voice=self.voice_tensor,
        speed=1, split_pattern=r'\n+'
        )
        # play the speech
        for i, (gs, ps, audio) in enumerate(generator):
            sd.play(audio, 24000)
            sd.wait()
        self.last_friction = friction
        return SpeechOutputInterface(speech_output=True)

    # def finalize(self):
    #     """
    #     Perform any necessary cleanup. This method is guaranteed
    #     to be called after the final `get_output` and run on the
    #     main thread.

    #     ***Nothing here
    #     """

    # def is_done(self) -> bool:
    #     """
    #     Return True if the demo should exit. This will
    #     always return False if not overridden.

    #     ***Nothing here
    #     """
    #     return False