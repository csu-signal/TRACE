import re
from typing import final

import cv2 as cv
import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    CameraCalibrationInterface,
    ColorImageInterface,
    CommonGroundInterface,
    FrictionOutputInterface,
    GazeConesInterface,
    GestureConesInterface,
    SelectedObjectsInterface,
    PlannerInterface,SpeechOutputInterface, UserInterface
)
from mmdemo.interfaces.data import Cone
from mmdemo.utils.coordinates import camera_3d_to_pixel


class Color:
    def __init__(self, name, color):
        self.name = name
        self.color = color


colors = [
    Color("red", (0, 0, 255)),
    Color("blue", (255, 0, 0)),
    Color("green", (19, 129, 51)),
    Color("purple", (128, 0, 128)),
    Color("yellow", (0, 215, 255)),
]

fontScales = [1.5, 1.5, 0.75, 0.5, 0.5]
fontThickness = [3, 3, 2, 2, 2]


@final
class UserFrame(BaseFeature[ColorImageInterface]):
    """
    Return the output frame used in the EMNLP Demo

    Input interfaces are `ColorImageInterface`, `GazeConesInterface`,
    `GestureConesInterface`, `SelectedObjectsInterface`,
    `CommonGroundInterface`

    Output interface is `Joe`
    """

    def __init__(
        self,
        speechoutput: BaseFeature[SpeechOutputInterface],
        friction: BaseFeature[FrictionOutputInterface]
    ):
        super().__init__(speechoutput, friction)
        # if plan is None:
        #     super().__init__(speechoutput, friction, common_ground ) # removed gaze
        #     if common_ground is None:
        # else:
        #     super().__init__(speechoutput, friction, common_ground, plan) # removed gaze

    def initialize(self):
        self.has_cgt_data = False
        self.last_plan = {"text": "", "color": (255, 255, 255)}
        

    def get_output(
        self,
        speech: SpeechOutputInterface,
        friction: FrictionOutputInterface,
        # common: CommonGroundInterface = None,
        # plan: PlannerInterface = None,
    ):
        # if (
        #     not common.is_new()
        #     or not speech.is_new()
        #     or not friction.is_new()
        # ):
        #     return None

        # ensure we are not modifying the color frame itself
        # output_frame = np.copy(color.frame)
        # output_frame = cv.cvtColor(output_frame, cv.COLOR_RGB2BGR)
        if speech.length > 0:
            output_frame = cv.imread(r"C:\Users\Multimodal_Demo\TRACE\mmdemo\features\speech_output\joe_assistant_idea.JPG")
        else:
            output_frame = cv.imread(r"C:\Users\Multimodal_Demo\TRACE\mmdemo\features\speech_output\joe_assistant.jpeg")



        if friction and friction.friction_statement != '':
            frictionStatements = self.get_dpip_friction_output(friction)
            fstate = frictionStatements#[0].split(":")[-1]
            x, y = (900, 300)
            text = fstate
            text = text.split()
            font = cv.FONT_HERSHEY_SIMPLEX
            font_scale = 2
            font_thickness = 5
            text_color_bg = (255,255,255)
            text_color =(0,0,0)
            text_size, _ = cv.getTextSize(str(text), font, font_scale, font_thickness)
            text_w, text_h = text_size

            max_chars = 30
            word = 0
            while(word<len(text)):
                text_row = ""
                while(word<len(text)):
                    if(len(text_row) + len(text[word])) < max_chars:
                        text_row += " " + text[word]
                        word += 1
                    else: break
                cv.putText(output_frame, str(text_row), (int(x), int(y + text_h + font_scale - 1)), font, font_scale, text_color, font_thickness, cv.LINE_AA)
                y += 75


        output_frame = cv.resize(output_frame, (900, 900))
        output_frame = cv.cvtColor(output_frame, cv.COLOR_BGR2RGB)

        return ColorImageInterface(frame=output_frame, frame_count=0)

    def get_dpip_friction_output(self,frictionout):
        friction = frictionout.friction_statement.split('\n')
        ranking = frictionout.ranking.split('\n')
        min = 10
        user = "Group"

        if len(friction) == 0:
            return ''
        
        #play a request for interruption
        if(ranking != ''):
            try:
                for rank in ranking:
                    a = rank.split(": ")
                    r = int(a[1])
                    if(r != 0 and r < min):
                        min = r
                        user = a[0]
            except Exception as e:
                user = "Group"
                print("Rank Error, defaulting to group")

        statements = []
        frictionStatement=''
    
        try:
            #if there's only one friction statement and it's length is greater than 4 read it
            if(len(friction) == 1):
                if(len(friction[0].split(' ')) > 4):
                    frictionStatement = f
            else:
                #if there are multiple statements find the value with the highest rank and track the variations in length and statements
                for f in friction:
                    if(f != ''):
                        statement = f.split(": ")[1]
                        statements.append(statement)
                        statementLength = len(statement.split(' '))
                        
                        if user in f:
                            # if the highest ranking friction statement isn't at least 4 words use the group
                            if(statementLength > 4):
                                frictionStatement = f

                            else:
                                frictionStatement = friction[-1] 

            #if all the statments are the same and more than 4 words, read the last one (group?)
            #if they are all the same and less than 4 words skip
            if(len(set(statements)) == 1):
                if(statementLength > 4):
                    frictionStatement = friction[-1]
                else:
                    frictionStatement = ''
                    
        except Exception as e:
            frictionStatement=''
            print("Friction Parsing Error")
            
        # opening = random.choice(os.listdir("C:/GitHub/TRACE/mmdemo/features/speech_output/audio"))
        # audio, samplerate = sf.read(fr"C:/GitHub/TRACE/mmdemo/features/speech_output/audio/{opening}")
        # sd.wait()
        # sd.play(audio, samplerate)
        # sd.wait()
        #generate speech, splits on newline
        #new friction and unqiue statements

        #Removing the user tag for user study (MB)
        frictionStatement = frictionStatement.replace(user," ").replace(":","")
        return frictionStatement