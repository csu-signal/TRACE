import time

from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    PoseEventInterface,
    GazeEventInterface,
    EngagementLevelInterface
)

@final
class EngagementLevel(BaseFeature[EngagementLevelInterface]):
    """
    This feature gathers all gaze and posture events and keeps track of these events
    It maintains a timer for each event
    If a event countdown time hits the upper bound, remove it from active event list

    Input interfaces are `PoseEventInterface` and `GazeEventInterface`

    Output interface is `EngagementLevelInterface`
    """
    def __init__(
        self,
        pose_event: BaseFeature[PoseEventInterface],
        gaze_event: BaseFeature[GazeEventInterface],
        live: bool,
        time_interval: float,
        frame_rate: int,
        gaze_positive_timer: float,
        gaze_negative_timer: float,
        posture_positive_timer: float,
        posture_negative_timer: float
    ):
        super().__init__(pose_event, gaze_event)
        #initialize all parameters
        self.live = live
        self.time_interval = time_interval
        self.frame_rate = frame_rate
        self.gaze_positive_timer = gaze_positive_timer
        self.gaze_negative_timer = gaze_negative_timer
        self.posture_positive_timer = posture_positive_timer
        self.posture_negative_timer = posture_negative_timer

    def initialize(self):
        #record the current time and the time of last frame
        self.time = 0
        self.last_frame_time = time.time()
        self.report_cooling_time = self.time_interval

        #the beginning engagement level is 2
        self.engagement_level = 2

        #timer for each event
        self.timers = {"gaze_positive":[], "gaze_negative":[], "posture_positive":[], "posture_negative":[]}

        file = open(f"./behavioral_engagement_level_log.csv", "w", encoding = "utf-8")
        file.write("time,active events,action,level\n")
        file.close()
    
    def get_output(
        self,
        pose_event: PoseEventInterface,
        gaze_event: GazeEventInterface
    ):
        #whether receive new information or not, update the current time
        current_time = time.time()
        if self.live:
            self.time += current_time - self.last_frame_time
            self.report_cooling_time -= current_time - self.last_frame_time
        else:
            self.time += 1 / self.frame_rate
            self.report_cooling_time -= 1 / self.frame_rate

        #update timer
        self.timer_update(current_time)

        if (not pose_event.is_new()
            and not gaze_event.is_new()):
            #if do not receive new information, keep the engagement level
            return EngagementLevelInterface(engagement_level = self.engagement_level)

        #if receive new gaze events, push them into the timer
        if gaze_event.is_new():
            for i in range(gaze_event.positive_event):
                self.timers["gaze_positive"].append(self.gaze_positive_timer)
            for i in range(gaze_event.negative_event):
                self.timers["gaze_negative"].append(self.gaze_negative_timer)
        
        #if receive new posture event, push them into the timer
        if pose_event.is_new():
            for i in range(pose_event.positive_event):
                self.timers["posture_positive"].append(self.posture_positive_timer)
            for i in range(pose_event.negative_event):
                self.timers["posture_negative"].append(self.posture_negative_timer)
        
        #update last frame time
        if self.live:
            self.last_frame_time = current_time

        #if it is not the time to update behavorial engagement level, report the old level
        #use 0.00001 here to mitigate the influence of accuracy of decimal
        if self.report_cooling_time > 0.00001:
            return EngagementLevelInterface(engagement_level = self.engagement_level)
        
        #the difference between the number of positive events and the number of negative events
        n = 0
        n += len(self.timers["gaze_positive"])
        n += len(self.timers["posture_positive"])
        n -= len(self.timers["gaze_negative"])
        n -= len(self.timers["posture_negative"])

        file = open(f"./behavioral_engagement_level_log.csv", "a", encoding = "utf-8")

        #the number of active positive events is more than the number of active negative events by at least 4
        if n >= 4:
            #update the behavioral engagement level and log it as a event
            if self.engagement_level < 3:
                self.engagement_level += 1
                file.write(str(self.time) + "," + \
                           str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                           str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                           "behavioral engagemental level +1," + str(self.engagement_level) + "\n")
            else:
                self.engagement_level = 3
                file.write(str(self.time) + "," + \
                           str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                           str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                           "none," + str(self.engagement_level) + "\n")
            #clear all events
            for timer in self.timers:
                self.timers[timer].clear()
        #the number of active negative events is more than the number of active positive events by at least 4
        elif n <= -4:
            #update the behavioral engagement level and log it as a event
            if self.engagement_level > 1:
                self.engagement_level -= 1
                file.write(str(self.time) + "," + \
                           str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                           str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                           "behavioral engagemental level -1," + str(self.engagement_level) + "\n")
            else:
                self.engagement_level = 1
                file.write(str(self.time) + "," + \
                           str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                           str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                           "none," + str(self.engagement_level) + "\n")
            #clear all events
            for timer in self.timers:
                self.timers[timer].clear()
        else:
            #log the current behavioral engagement level
            file.write(str(self.time) + "," + \
                           str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                           str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                           "none," + str(self.engagement_level) + "\n")

        file.close()
        
        #reset the cooling time
        self.report_cooling_time = self.time_interval
        
        return EngagementLevelInterface(engagement_level = self.engagement_level)

    def timer_update(self, current_time):
        file = open(f"./behavioral_engagement_level_log.csv", "a", encoding = "utf-8")
        for timer in self.timers:
            #subtract time from each timer
            for i in range(len(self.timers[timer])):
                self.timers[timer][i] -= current_time - self.last_frame_time if self.live else 1 / self.frame_rate
            if timer == "gaze_positive":
                #for a kind of event, if the timer hits 0, remove from the queue
                #count how many events are no more active and log the information
                #use 0.00001 here to mitigate the influence of accuracy of decimal
                count = 0
                while len(self.timers[timer]) > 0 and self.timers[timer][0] <= 0.00001:
                    count += 1
                    self.timers[timer].pop(0)
                if count > 0:
                    file.write(str(self.time) + "," + \
                               str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                               str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                               str(count) + " gaze positive events are no more active," + str(self.engagement_level) + "\n")
            elif timer == "gaze_negative":
                count = 0
                while len(self.timers[timer]) > 0 and self.timers[timer][0] <= 0.00001:
                    count += 1
                    self.timers[timer].pop(0)
                if count > 0:
                    file.write(str(self.time) + "," + \
                               str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                               str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                               str(count) + " gaze negative events are no more active," + str(self.engagement_level) + "\n")
            elif timer == "posture_positive":
                count = 0
                while len(self.timers[timer]) > 0 and self.timers[timer][0] <= 0.00001:
                    count += 1
                    self.timers[timer].pop(0)
                if count > 0:
                    file.write(str(self.time) + "," + \
                               str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                               str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                               str(count) + " posture positive events are no more active," + str(self.engagement_level) + "\n")
            else:
                count = 0
                while len(self.timers[timer]) > 0 and self.timers[timer][0] <= 0.00001:
                    count += 1
                    self.timers[timer].pop(0)
                if count > 0:
                    file.write(str(self.time) + "," + \
                               str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                               str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                               str(count) + " posture negative events are no more active," + str(self.engagement_level) + "\n")
        file.close()