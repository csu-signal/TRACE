from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    PoseInterface,
    PoseEventInterface
)

@final
class PoseEvent(BaseFeature[PoseEventInterface]):
    """
    This feature is used to detect events happening in leaning in/out
    It maintains queues for different participants
    """
    def __init__(self,
        po: BaseFeature[PoseInterface],
        frame_rate: int,
        history: int,
        positive_event_time: int,
        negative_event_time: int,
        leanout_count_time: int,
        smooth_frame: int
    ) -> None:
        super().__init__(po)
        self.frame_rate = frame_rate
        self.history = int(history * frame_rate)
        self.positive_event_time = int(positive_event_time * frame_rate)
        self.negative_event_time = int(negative_event_time * frame_rate)
        self.leanout_count_time = int(leanout_count_time * frame_rate)
        self.smooth_frame = smooth_frame
    
    def initialize(
        self
    ) -> None:
        #Use dictionary to store queue for different participants
        #The id of participants may be different between this feature and the gesture feature
        self.queue = {}
        self.count_frame = 0
        self.positive_cooling_time = {}
        self.negative_cooling_time = {}

    def get_output(
        self,
        po: PoseInterface
    ) -> PoseEventInterface | None:
        #Count a new frame
        self.count_frame += 1

        #If the queue is full, remove the first item
        for body in self.queue:
            if len(self.queue[body]) >= self.history:
                self.queue[body].pop(0)

        #no new information received, append None into the queue
        if not po.is_new():
            for body in self.queue:
                for i in range(self.smooth_frame):
                    if self.queue[body][-1 - i][1] is True:
                        self.queue[body].append((self.queue[body][-1 - i][0], False))
                        break
                else:
                    self.queue[body].append((None, False))
        else:
            #If receive new information
            #Put leaning in/out information into the queue
            for body, pose in po.poses:
                if body not in self.positive_cooling_time:
                    self.positive_cooling_time[body] = self.positive_event_time
                    self.negative_cooling_time[body] = self.negative_event_time
                if body in self.queue:
                    self.queue[body].append((pose, True))
                else:
                    self.queue[body] = [(pose, True)]
            
            for body in self.queue:
                if body not in [i for i, _ in po.poses]:
                    for i in range(self.smooth_frame):
                        if self.queue[body][-1 - i][1] is True:
                            self.queue[body].append((self.queue[body][-1 - i][0], False))
                            break
                    else:
                        self.queue[body].append((None, False))

        for body in self.negative_cooling_time:
            self.negative_cooling_time[body] -= 1 if self.negative_cooling_time[body] > 0 else 0
        for body in self.positive_cooling_time:
            self.positive_cooling_time[body] -= 1 if self.positive_cooling_time[body] > 0 else 0
        
        pe, ne = self.event_decision()

        return PoseEventInterface(positive_event = pe, negative_event = ne)
    
    def event_decision(self):
        positive_event = 0
        negative_event = 0

        file = open(f"./event_log.csv", "a", encoding = "utf-8")
        for body in self.queue:
            if len(self.queue[body]) >= self.leanout_count_time and self.negative_cooling_time[body] == 0:
                if all(i[0] != "leaning in" for i in self.queue[body][-self.leanout_count_time:]):
                    negative_event += 1
                    file.write(str(self.count_frame / self.frame_rate) + "," + str(self.count_frame) + "," + str(body) + " negative posture event\n")
                    self.negative_cooling_time[body] = self.negative_event_time
                    self.positive_cooling_time[body] = self.positive_event_time
                else:
                    if self.positive_cooling_time[body] == 0:
                        positive_event += 1
                        file.write(str(self.count_frame / self.frame_rate) + "," + str(self.count_frame) + "," + str(body) + " positive posture event\n")
                        self.positive_cooling_time[body] = self.positive_event_time
    
        file.close()

        return positive_event, negative_event

    """
    def event_decision(self):
        positive_event = 0
        negative_event = 0

        file = open(f"./event_log.csv", "a", encoding = "utf-8")
        for body in self.queue:
            if self.individual_cooling_time[body] == 0:
                individual_ne_event = 0
                count_change = 0
                count_buffer = 0
                current_stage = "leaning in" if [i[0] for i in self.queue[body][0:self.posture_event_buffer]].count("leaning in") >= \
                                [i[0] for i in self.queue[body][0:self.posture_event_buffer]].count("leaning out") else "leaning out"
                i = self.posture_event_buffer
                while i < len(self.queue[body]):
                    if current_stage != self.queue[body][i][0]:
                        count_buffer += 1
                    else:
                        count_buffer = 0
                    if count_buffer >= self.posture_event_buffer:
                        count_change += 1
                        current_stage = self.queue[body][i][0]
                        count_buffer = 0
                    i += 1
                
                if count_change >= self.posture_change:
                    file.write(str(self.count_frame / self.frame_rate) + "," + str(self.count_frame) + "," + str(body) + " negative posture event: change posture more than 3 times\n")
                    negative_event += 1
                    individual_ne_event += 1
                
                count_negative = 0
                count_buffer = 0
                for i in range(len(self.queue[body])):
                    if self.queue[body][i][0] != "leaning in":
                        count_negative += 1
                        count_buffer = 0
                    else:
                        count_negative += 1
                        count_buffer += 1
                    if count_buffer >= self.posture_event_buffer:
                        count_negative = 0
                    if count_negative >= self.leanout_time:
                        negative_event += 1
                        individual_ne_event += 1
                        file.write(str(self.count_frame / self.frame_rate) + "," + str(self.count_frame) + "," + str(body) + " negative posture event: lean out for more than 10 seconds\n")
                        break

                if individual_ne_event == 0:
                    file.write(str(self.count_frame / self.frame_rate) + "," + str(self.count_frame) + "," + str(body) + " positive posture event\n")
                    positive_event += 1
                    
                self.individual_cooling_time[body] = self.history
        
        file.close()

        return positive_event, negative_event
        """