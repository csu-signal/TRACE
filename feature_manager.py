from tkinter import Tk, IntVar, Checkbutton
from featureModules import (AsrFeature, BaseDevice, GazeBodyTrackingFeature,
                            GazeFeature, GestureFeature, MicDevice,
                            MoveFeature, ObjectFeature, PoseFeature,
                            PrerecordedDevice, PropExtractFeature, rec_common_ground, DenseParaphrasingFeature, CommonGroundFeature)
from input_profile import BaseProfile
from logger import Logger
import cv2 as cv

class FeatureManager:
    def __init__(self, profile: BaseProfile, output_dir=None):
        self.output_dir = output_dir
        self.profile = profile

        self.root = Tk()
        # self.root.geometry('350x200')
        self.root.title("Output Options")

        self.vars = {
                "gesture": IntVar(value=1),
                "objects": IntVar(value=1),
                "gaze": IntVar(value=1),
                "asr": IntVar(value=1),
                "dense paraphrasing": IntVar(value=1),
                "pose": IntVar(value=0),
                "prop": IntVar(value=1),
                "move": IntVar(value=1),
                "common ground": IntVar(value=1),
                }

        self._create_buttons()

        shift = 7 # TODO what is this?
        self.gaze = GazeBodyTrackingFeature(shift, log_dir=self.output_dir)
        self.gesture = GestureFeature(shift, log_dir=self.output_dir)
        self.objects = ObjectFeature(log_dir=self.output_dir)
        self.pose = PoseFeature(log_dir=self.output_dir)
        self.asr = AsrFeature(self.profile.get_audio_devices(), n_processors=1, log_dir=self.output_dir)
        self.dense_paraphrasing = DenseParaphrasingFeature(log_dir=self.output_dir)
        self.prop = PropExtractFeature(log_dir=self.output_dir)
        self.move = MoveFeature(log_dir=self.output_dir)
        self.common_ground = CommonGroundFeature(log_dir=self.output_dir)

        if self.output_dir is not None:
            self.error_log = Logger(file=self.output_dir / "errors.txt", stdout=True)
            self.summary_log = Logger(file=self.output_dir / "summary.txt", stdout=True)
        else:
            self.error_log = Logger(stdout=True)
            self.summary_log = Logger(stdout=True)

        self.error_log.clear()
        self.summary_log.clear()

    def _create_buttons(self):
        for text,var in self.vars.items():
            Checkbutton(self.root, text=text, variable=var, onvalue=1, offvalue=0, height=2, width=10).pack()

    def _should_process(self, var):
        return self.vars[var].get()

    def processFrame(self, output_frame, framergb, depth, bodies, rotation, translation, cameraMatrix, distortion, frame_count):
        device_id = 0
        h,w,_ = output_frame.shape

        self.root.update()

        # run features
        blockStatus = {}
        blocks = []

        if(self._should_process("objects")):
            blocks = self.objects.processFrame(framergb, frame_count)

        if(self._should_process("pose")):
            self.pose.processFrame(bodies, output_frame, frame_count, False)

        try:
            if(self._should_process("gaze")):
                self.gaze.processFrame( bodies, w, h, rotation, translation, cameraMatrix, distortion, output_frame, framergb, depth, blocks, blockStatus, frame_count)
        except:
            pass
        
        if(self._should_process("gesture")):
             self.gesture.processFrame(device_id, bodies, w, h, rotation, translation, cameraMatrix, distortion, output_frame, framergb, depth, blocks, blockStatus, frame_count, False)

        new_utterances = []
        if(self._should_process("asr")):
            new_utterances = self.asr.processFrame(output_frame, frame_count, False)

        if self._should_process("dense paraphrasing"):
            self.dense_paraphrasing.processFrame(output_frame, new_utterances, self.asr.utterance_lookup, self.gesture.blockCache, frame_count)

        try:
            if(self._should_process("prop")):
                self.prop.processFrame(output_frame, new_utterances, self.dense_paraphrasing.paraphrased_utterance_lookup, frame_count, False)
        except Exception as e:
            self.error_log.append(f"Frame {frame_count}\nProp extractor\n{new_utterances}\n{str(e)}\n\n")

        if(self._should_process("move")):
            self.move.processFrame(output_frame, new_utterances, self.dense_paraphrasing.paraphrased_utterance_lookup, frame_count, False)

        if self._should_process("common ground"):
            self.common_ground.processFrame(output_frame, new_utterances, self.prop.prop_lookup, self.move.move_lookup, frame_count)


        cv.putText(output_frame, "FRAME:" + str(frame_count), (50,50), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)
        #cv.putText(frame, "DEVICE:" + str(int(device_id)), (50,100), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv.LINE_AA)

        self.update_summary(new_utterances, frame_count)

    def finalize(self):
        self.asr.done.value = True

    def update_summary(self, new_utterances, frame_count):
        for i in new_utterances:
            utterance = self.dense_paraphrasing.paraphrased_utterance_lookup[i]
            prop = self.prop.prop_lookup[i]
            move = self.move.move_lookup[i]

            update = ""
            update += "FRAME: " + str(frame_count) + "\n"
            update += "E bank\n"
            update += str(self.common_ground.closure_rules.ebank) + "\n"
            update += "F bank\n"
            update += str(self.common_ground.closure_rules.fbank) + "\n"
            if prop.prop == "no prop":
                update += f"{utterance.speaker_id}: {utterance.text} ({self.common_ground.most_recent_prop}), {move.move}\n\n"
            else:
                update += f"{utterance.speaker_id}: {utterance.text} => {prop.prop}, {move.move}\n\n"

            self.summary_log.append(update)
