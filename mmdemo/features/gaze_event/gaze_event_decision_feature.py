import time

from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    GazeSelectionInterface,
    GazeEventInterface
)

@final
class GazeEvent(BaseFeature[GazeEventInterface]):
    """
    The feature that decides and logs gaze events for each individual and the group

    Input interface is `GazeSelectionInterface`

    Output interface is `GazeEventInterface`
    """
    def __init__(self,
        gs: BaseFeature[GazeSelectionInterface],
        live: bool,
        frame_rate: int,
        history: float,
        individual_event_time: float,
        group_event_time: float,
        gaze_beginning_buffer: float,
        gaze_lookaway_buffer: float,
        smooth_frame: float,
        speaker: str
    ) -> None:
        super().__init__(gs)
        #initialize all parameters
        self.live = live
        self.frame_rate = frame_rate
        self.history = history
        self.individual_event_time = individual_event_time
        self.group_event_time = group_event_time
        self.gaze_beginning_buffer = gaze_beginning_buffer
        self.gaze_lookaway_buffer = gaze_lookaway_buffer
        self.smooth_frame = smooth_frame
        self.speaker = speaker
    
    def initialize(self) -> None:
        #initialize queue used to track different participant 
        self.queue = {}

        #record the current time and the time of last frame
        self.time = 0
        self.last_frame_time = time.time()

        #initialize cooling time which is the time interval to report gaze event for each participant and the group
        self.individual_cooling_time = {}
        self.group_cooling_time = self.group_event_time

        file = open(f"./event_log.csv", "w", encoding = "utf-8")
        file.write("time,event\n")
        file.close()
    
    def get_output(
        self,
        gs: GazeSelectionInterface,
    ) -> GazeEventInterface | None:
        #whether receive new input or not, update the current time
        if self.live:
            current_time = time.time()
            self.time += current_time - self.last_frame_time
        else:
            self.time += 1 / self.frame_rate

        #remove old information from queue for each participant if it exceeds the history window
        #each element in the queue is a tuple (gaze information, time)
        for body in self.queue:
            while self.queue[body][-1][1] - self.queue[body][0][1] >= self.history:
                self.queue[body].pop(0)

        #update queue
        self.update_queue(gs)

        #update the cooling time
        #if it is less than 0, set it to 0
        if self.live:
            self.group_cooling_time -= current_time - self.last_frame_time if self.group_cooling_time > 0 else 0
            for body in self.individual_cooling_time:
                self.individual_cooling_time[body] -= current_time - self.last_frame_time if self.individual_cooling_time[body] > 0 else 0
        else:
            self.group_cooling_time -= 1 / self.frame_rate if self.group_cooling_time > 0 else 0
            for body in self.individual_cooling_time:
                self.individual_cooling_time[body] -= 1 / self.frame_rate if self.individual_cooling_time[body] > 0 else 0

        #for the current frame, compute the number of positive events and negative events
        pe, ne = self.event_decision(self.speaker)

        #update last frame time
        if self.live:
            self.last_frame_time = current_time

        return GazeEventInterface(positive_event = pe, negative_event = ne)
    
    def update_queue(self, gs):
        #if no new input, push None into the queue
        #in the queue, each gaze information is a tuple (selected participant by gaze, whether it is newly output information)
        #for example ("P2", False) means that at this point, participant P2 is focused
        #however, it is not a newly received information
        #it is copied from the previous informtion to smooth frame 
        if not gs.is_new():
            for body in self.queue:
                #use history information to smooth frames
                i = len(self.queue[body]) - 1
                while i >= 0 and self.time - self.queue[body][i][1] <= self.smooth_frame:
                    if self.queue[body][i][0][1] is True:
                        self.queue[body].append(((self.queue[body][i][0][0], False), self.time))
                        break
                    i -= 1
                else:
                    self.queue[body].append(((None, False), self.time))
        else:
            #if there is new information, push it into the queue
            for i in range(len(gs.selection)):
                if gs.selection[i][0] not in self.individual_cooling_time:
                    self.individual_cooling_time[gs.selection[i][0]] = self.individual_event_time
                if gs.selection[i][0] in self.queue:
                    self.queue[gs.selection[i][0]].append(((gs.selection[i][1], True), self.time))
                else:
                    self.queue[gs.selection[i][0]] = [((gs.selection[i][1], True), self.time)]

            #if someone is not in the received feature, treat it as none information and do the same operation
            for body in self.queue:
                i = len(self.queue[body]) - 1
                while i >= 0 and self.time - self.queue[body][i][1] <= self.smooth_frame:
                    if self.queue[body][i][0][1] is True:
                        self.queue[body].append(((self.queue[body][i][0][0], False), self.time))
                        break
                    i -= 1
                else:
                    self.queue[body].append(((None, False), self.time))
    
    def check_gaze_selection(self, queue:list[tuple[tuple[str | None, bool], float]], time:float, speaker:str):
        """
        This function is used to check whether a participant looked at the speak in the previous several seconds

        Input is `queue` the gaze history of a participant, `time` the time interval we check and the `speaker`

        Output is a boolean value indicating whether the participant is looking at the speaker or None 
        """
        #find the beginning index
        begin = 0
        while begin < len(queue) - 1:
            if queue[-1][1] - queue[begin][1] >= time and queue[-1][1] - queue[begin + 1][1] < time:
                break
            begin += 1

        #count positive time, negative time, beginning buffer time, look away buffer time
        count_positive = 0
        count_negative = 0
        count_beginning = 0
        count_lookaway = 0
        
        #decide whether the participant is looking at the speaker in the given time interval
        for i in range(begin + 1, len(queue)):
            #update the time by whether the participant looks at the speaker
            if queue[i][0][0] == speaker:
                count_positive += queue[i][1] - queue[i - 1][1]
                count_beginning += queue[i][1] - queue[i - 1][1]
                count_lookaway = 0
            else:
                count_negative += queue[i][1] - queue[i - 1][1]
                count_lookaway += queue[i][1] - queue[i - 1][1]
                count_beginning = 0
            
            #if a participant looks at the speaker more than beginning buffer, treat it as looking at the speaker
            #set the negative time count to 0
            if count_beginning >= self.gaze_beginning_buffer:
                count_negative = 0
            else:
                count_negative += queue[i][1] - queue[i - 1][1]
            
            if count_lookaway >= self.gaze_lookaway_buffer:
                count_positive = 0
            else:
                count_positive += queue[i][1] - queue[i - 1][1]
            
            #if look away or look at the speaker for more than threshold
            #report it as a negative / positive event
            if count_positive >= self.individual_event_time:
                return True
            
            if count_negative >= self.individual_event_time:
                return False

        return None

    def event_decision(self, speaker: str):
        """
        This function is used to count how many positive events and negative events at the given time

        Input is the `speaker` (from "P1", "P2", "P3")
        """

        positive_event = 0
        negative_event = 0

        file = open(f"./event_log.csv", "a", encoding = "utf-8")
        for body in self.queue:
            #for a participant other than the speaker, if the cooling time is less than 0, compute positive events and negative events
            #use 0.00001 here to mitigate the influence of accuracy of decimal
            if body != speaker and self.individual_cooling_time[body] <= 0.00001:
                event = self.check_gaze_selection(self.queue[body], self.individual_event_time, speaker)

                #log the event
                if event is True:
                    positive_event += 1
                    file.write(str(self.time) + ","  + str(body) + " individual positive gaze event\n")
                    self.individual_cooling_time[body] = self.individual_event_time
                elif event is False:
                    negative_event += 1
                    file.write(str(self.time) + "," + str(body) + " individual negative gaze event\n")
                    self.individual_cooling_time[body] = self.individual_event_time

        #if the group cooling time becomes less than 0, compute positive events and negative events of the group
        #use 0.00001 here to mitigate the influence of accuracy of decimal
        if self.group_cooling_time <= 0.00001:
            group_result = []
            #check gazes of all participants except the speaker
            for body in self.queue:
                if body != speaker:
                    group_result.append(self.check_gaze_selection(self.queue[body], self.group_event_time, speaker))

            #log the event
            if all(i is True for i in group_result):
                positive_event += 1
                file.write(str(self.time) + ","  + "group positive gaze event\n")
                self.group_cooling_time = self.group_event_time
            elif all(i is False for i in group_result):
                negative_event += 1
                file.write(str(self.time) + "," + "group negative gaze event\n")
                self.group_cooling_time = self.group_event_time

        file.close()

        return positive_event, negative_event