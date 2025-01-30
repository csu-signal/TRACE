from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    GazeSelectionInterface,
    GazeEventInterface
)

@final
class GazeEvent(BaseFeature[GazeEventInterface]):
    """
    This feature is used to decide whether there happens
    positive or negative event in participant's gaze
    Queue is introduced to store historical information
    Sampling interval is used to set how frequently we make the dicision
    Window length is the time interval we make decisions on (length of queue)
    """
    def __init__(self,
        gs: BaseFeature[GazeSelectionInterface],
        frame_rate: int,
        history: int,
        individual_event_time: int,
        group_event_time: int,
        gaze_beginning_buffer: int,
        gaze_lookaway_buffer: int,
        smooth_frame: int,
        speaker: str
    ) -> None:
        super().__init__(gs)
        #initialize time interval, window size and the frame rate
        self.frame_rate = frame_rate
        self.history = int(history * frame_rate)
        self.individual_event_time = int(individual_event_time * frame_rate)
        self.group_event_time = int(group_event_time * frame_rate)
        self.gaze_beginning_buffer = int(gaze_beginning_buffer * frame_rate)
        self.gaze_lookaway_buffer = int(gaze_lookaway_buffer * frame_rate)
        self.smooth_frame = smooth_frame
        self.speaker = speaker
    
    def initialize(self) -> None:
        #Initizalize queue in the form of dict in python
        #the key and value is participant id : action list[]
        #start frame count from 0
        self.queue = {}
        self.count_frame = 0
        self.individual_cooling_time = {}
        self.group_cooling_time = self.group_event_time
        file = open(f"./event_log.csv", "w", encoding = "utf-8")
        file.write("time,frame,event\n")
        file.close()
    
    def get_output(
        self,
        gs: GazeSelectionInterface,
    ) -> GazeEventInterface | None:
        #Whether receive new input or not, frame count plus 1
        self.count_frame += 1

        #Delete old information from queue for each participant if it exceeds the individual history window size
        for body in self.queue:
            if len(self.queue[body]) >= self.history:
                self.queue[body].pop(0)

        #If no new input, push None into the queue
        if not gs.is_new():
            for body in self.queue:
                for i in range(self.smooth_frame):
                    if self.queue[body][-1 - i][1] is True:
                        self.queue[body].append((self.queue[body][-1 - i][0], False))
                        break
                else:
                    self.queue[body].append((None, False))
        else:
            #If there is new information, we push it into the queue
            for i in range(len(gs.selection)):
                if gs.selection[i][0] not in self.individual_cooling_time:
                    self.individual_cooling_time[gs.selection[i][0]] = self.individual_event_time
                if gs.selection[i][0] in self.queue:
                    self.queue[gs.selection[i][0]].append((gs.selection[i][1], True))
                else:
                    self.queue[gs.selection[i][0]] = [(gs.selection[i][1], True)]

            #If someone is not recorded, we push None into the queue
            for body in self.queue:
                if body not in [i[0] for i in gs.selection]:
                    for i in range(self.smooth_frame):
                        if self.queue[body][-1 - i][1] is True:
                            self.queue[body].append((self.queue[body][-1 - i][0], False))
                            break
                    else:
                        self.queue[body].append((None, False))

        self.group_cooling_time -= 1 if self.group_cooling_time > 0 else 0
        for body in self.individual_cooling_time:
            self.individual_cooling_time[body] -= 1 if self.individual_cooling_time[body] > 0 else 0

        pe, ne = self.check_speaker_selection(self.speaker)
        
        return GazeEventInterface(positive_event = pe, negative_event = ne)
    
    def check_speaker_selection(self, speaker):
        positive_event = 0
        negative_event = 0

        file = open(f"./event_log.csv", "a", encoding = "utf-8")
        for body in self.queue:
            if body != speaker and self.individual_cooling_time[body] == 0 and len(self.queue[body]) >= self.individual_event_time:
                count_positive = 0
                count_negative = 0
                count_beginning = 0
                count_lookaway = 0
                for i in self.queue[body][-self.individual_event_time:]:
                    if i[0] != speaker:
                        count_negative += 1
                        count_beginning = 0
                        count_lookaway += 1
                    else:
                        count_positive += 1
                        count_beginning += 1
                        count_lookaway = 0
                    if count_beginning >= self.gaze_beginning_buffer:
                        count_negative = 0
                    else:
                        count_negative += 1
                    if count_lookaway >= self.gaze_lookaway_buffer:
                        count_positive = 0
                    else:
                        count_positive += 1
                    if count_negative >= self.individual_event_time:
                        negative_event += 1
                        file.write(str(self.count_frame / self.frame_rate) + "," + str(self.count_frame) + "," + str(body) + " individual negative gaze event\n")
                        self.individual_cooling_time[body] = self.individual_event_time
                        break
                    if count_positive >= self.individual_event_time:
                        positive_event += 1
                        file.write(str(self.count_frame / self.frame_rate) + "," + str(self.count_frame) + "," + str(body) + " individual positive gaze event\n")
                        self.individual_cooling_time[body] = self.individual_event_time
                        break

        if self.group_cooling_time == 0 and all(len(self.queue[body]) >= self.group_event_time for body in self.queue):
            count_positive = 0
            count_lookaway = {body:0 for body in self.queue if body != speaker}
            count_negative = 0
            count_beginning = {body:0 for body in self.queue if body != speaker}
            for i in range(self.group_event_time):
                gaze_target = {body:self.queue[body][len(self.queue[body]) - self.group_event_time + i][0] for body in self.queue if body != speaker}
                if all(j == speaker for j in gaze_target.values()):
                    count_positive += 1
                    for body in count_lookaway:
                        count_lookaway[body] = 0
                        count_beginning[body] += 1
                elif all(j != speaker for j in gaze_target.values()):
                    count_negative += 1
                    for body in count_lookaway:
                        count_lookaway[body] += 1
                        count_beginning[body] = 0
                else:
                    for body in gaze_target:
                        if gaze_target[body] == speaker:
                            count_beginning[body] += 1
                            count_lookaway[body] = 0
                        else:
                            count_beginning[body] = 0
                            count_lookaway[body] += 1
                
                if any(j >= self.gaze_lookaway_buffer for j in count_lookaway.values()):
                    count_positive = 0
                else:
                    count_positive += 1
                if any(j >= self.gaze_beginning_buffer for j in count_beginning.values()):
                    count_negative = 0
                else:
                    count_negative += 1

                if count_positive >= self.group_event_time:
                    positive_event += 1
                    file.write(str(self.count_frame / self.frame_rate) + "," + str(self.count_frame) + "," + "group positive gaze event\n")
                    self.group_cooling_time = self.group_event_time
                    break
                if count_negative >= self.group_event_time:
                    negative_event += 1
                    file.write(str(self.count_frame / self.frame_rate) + "," + str(self.count_frame) + "," + "group negative gaze event\n")
                    self.group_cooling_time = self.group_event_time
                    break

        file.close()

        return positive_event, negative_event