from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    ColorImageInterface,
    PoseEventInterface,
    GazeEventInterface,
    EngagementLevelInterface,
)

@final
class EngagementLevel(BaseFeature[ColorImageInterface]):
    def __init__(
        self,
        pose_event: BaseFeature[PoseEventInterface],
        gaze_event: BaseFeature[GazeEventInterface],
        time_interval: int,
        frame_rate: int,
        gaze_positive_timer: int,
        gaze_negative_timer: int,
        posture_positive_timer: int,
        posture_negative_timer: int,
    ):
        super().__init__(pose_event, gaze_event)
        self.time_interval = time_interval
        self.frame_rate = frame_rate
        self.gaze_positive_timer = gaze_positive_timer
        self.gaze_negative_timer = gaze_negative_timer
        self.posture_positive_timer = posture_positive_timer
        self.posture_negative_timer = posture_negative_timer

    def initialize(self):
        self.frame_count = 0
        self.engagement_level = 2
        self.timers = {"gaze_positive":[], "gaze_negative":[], "posture_positive":[], "posture_negative":[]}
        file = open(f"./behavioral_engagement_level_log.csv", "w", encoding = "utf-8")
        file.write("time,frame,active events,action,level\n")
        file.close()
    
    def get_output(
        self,
        pose_event: PoseEventInterface,
        gaze_event: GazeEventInterface
    ):
        self.frame_count += 1
        self.timer_update()

        if (not pose_event.is_new()
            and not gaze_event.is_new()):
            return EngagementLevelInterface(engagement_level = self.engagement_level)

        if gaze_event.is_new():
            for i in range(gaze_event.positive_event):
                self.timers["gaze_positive"].append(0)
            for i in range(gaze_event.negative_event):
                self.timers["gaze_negative"].append(0)
        
        if pose_event.is_new():
            for i in range(pose_event.positive_event):
                self.timers["posture_positive"].append(0)
            for i in range(pose_event.negative_event):
                self.timers["posture_negative"].append(0)

        if self.frame_count % (int(self.time_interval * self.frame_rate)) != 0:
            return EngagementLevelInterface(engagement_level = self.engagement_level)
        
        n = 0
        n += len(self.timers["gaze_positive"])
        n += len(self.timers["posture_positive"])
        n -= len(self.timers["gaze_negative"])
        n -= len(self.timers["posture_negative"])

        file = open(f"./behavioral_engagement_level_log.csv", "a", encoding = "utf-8")

        if n >= 4:
            if self.engagement_level < 3:
                self.engagement_level += 1
                file.write(str(self.frame_count / self.frame_rate) + "," + str(self.frame_count) + "," + \
                           str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                           str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                           "behavioral engagemental level +1," + str(self.engagement_level) + "\n")
            else:
                self.engagement_level = 3
                file.write(str(self.frame_count / self.frame_rate) + "," + str(self.frame_count) + "," + \
                           str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                           str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                           "none," + str(self.engagement_level) + "\n")
            for timer in self.timers:
                self.timers[timer].clear()
        elif n <= -4:
            if self.engagement_level > 1:
                self.engagement_level -= 1
                file.write(str(self.frame_count / self.frame_rate) + "," + str(self.frame_count) + "," + \
                           str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                           str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                           "behavioral engagemental level -1," + str(self.engagement_level) + "\n")
            else:
                self.engagement_level = 1
                file.write(str(self.frame_count / self.frame_rate) + "," + str(self.frame_count) + "," + \
                           str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                           str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                           "none," + str(self.engagement_level) + "\n")
            for timer in self.timers:
                self.timers[timer].clear()
        else:
            file.write(str(self.frame_count / self.frame_rate) + "," + str(self.frame_count) + "," + \
                           str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                           str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                           "none," + str(self.engagement_level) + "\n")

        file.close()
        
        return EngagementLevelInterface(engagement_level = self.engagement_level)

    def timer_update(self):
        file = open(f"./behavioral_engagement_level_log.csv", "a", encoding = "utf-8")
        for timer in self.timers:
            for i in range(len(self.timers[timer])):
                self.timers[timer][i] += 1
            if timer == "gaze_positive":
                count = 0
                while len(self.timers[timer]) > 0 and self.timers[timer][0] >= self.gaze_positive_timer * self.frame_rate:
                    count += 1
                    self.timers[timer].pop(0)
                if count > 0:
                    file.write(str(self.frame_count / self.frame_rate) + "," + str(self.frame_count) + "," + \
                               str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                               str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                               str(count) + " gaze positive events are no more active," + str(self.engagement_level) + "\n")
            elif timer == "gaze_negative":
                count = 0
                while len(self.timers[timer]) > 0 and self.timers[timer][0] >= self.gaze_negative_timer * self.frame_rate:
                    count += 1
                    self.timers[timer].pop(0)
                if count > 0:
                    file.write(str(self.frame_count / self.frame_rate) + "," + str(self.frame_count) + "," + \
                               str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                               str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                               str(count) + " gaze negative events are no more active," + str(self.engagement_level) + "\n")
            elif timer == "posture_positive":
                count = 0
                while len(self.timers[timer]) > 0 and self.timers[timer][0] >= self.posture_positive_timer * self.frame_rate:
                    count += 1
                    self.timers[timer].pop(0)
                if count > 0:
                    file.write(str(self.frame_count / self.frame_rate) + "," + str(self.frame_count) + "," + \
                               str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                               str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                               str(count) + " posture positive events are no more active," + str(self.engagement_level) + "\n")
            else:
                count = 0
                while len(self.timers[timer]) > 0 and self.timers[timer][0] >= self.posture_negative_timer * self.frame_rate:
                    count += 1
                    self.timers[timer].pop(0)
                if count > 0:
                    file.write(str(self.frame_count / self.frame_rate) + "," + str(self.frame_count) + "," + \
                               str(len(self.timers["gaze_positive"]) + len(self.timers["posture_positive"])) + " positive events" + \
                               str(len(self.timers["gaze_negative"]) + len(self.timers["posture_negative"])) + " negative events," + \
                               str(count) + " posture negative events are no more active," + str(self.engagement_level) + "\n")
        file.close()