import time
from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import PoseEventInterface, PoseInterface


@final
class PoseEvent(BaseFeature[PoseEventInterface]):
    """
    The feature that decides and logs posture events for each individual

    Input interface is `PoseInterface`

    Output interface is `PoseEventInterface`
    """

    def __init__(
        self,
        po: BaseFeature[PoseInterface],
        live: bool,
        frame_rate: int,
        history: float,
        positive_event_time: float,
        negative_event_time: float,
        leanout_count_time: float,
        smooth_frame: float,
    ) -> None:
        super().__init__(po)
        # initialize all parameters
        self.live = live
        self.frame_rate = frame_rate
        self.history = history
        self.positive_event_time = positive_event_time
        self.negative_event_time = negative_event_time
        self.leanout_count_time = leanout_count_time
        self.smooth_frame = smooth_frame

    def initialize(self) -> None:
        # initialize queue used to track different participant
        self.queue = {}

        # record the current time and the time of last frame
        self.time = 0
        self.last_frame_time = time.time()

        # initialize cooling time which is the time interval to report posture event for each participant
        self.positive_cooling_time = {}
        self.negative_cooling_time = {}

    def get_output(self, po: PoseInterface) -> PoseEventInterface | None:
        # whether receive new input or not, update the current time
        if self.live:
            current_time = time.time()
            self.time += current_time - self.last_frame_time
        else:
            self.time += 1 / self.frame_rate

        # remove old information from queue for each participant if it exceeds the history window
        # each element in the queue is a tuple (pose information, time)
        for body in self.queue:
            while self.queue[body][-1][1] - self.queue[body][0][1] >= self.history:
                self.queue[body].pop(0)

        # update queue
        self.update_queue(po)

        # update the cooling time
        # if it is less than 0, set it to 0
        if self.live:
            for body in self.positive_cooling_time:
                self.positive_cooling_time[body] -= (
                    current_time - self.last_frame_time
                    if self.positive_cooling_time[body] > 0
                    else 0
                )
            for body in self.negative_cooling_time:
                self.negative_cooling_time[body] -= (
                    current_time - self.last_frame_time
                    if self.negative_cooling_time[body] > 0
                    else 0
                )
        else:
            for body in self.positive_cooling_time:
                self.positive_cooling_time[body] -= (
                    1 / self.frame_rate if self.positive_cooling_time[body] > 0 else 0
                )
            for body in self.negative_cooling_time:
                self.negative_cooling_time[body] -= (
                    1 / self.frame_rate if self.negative_cooling_time[body] > 0 else 0
                )

        # for the current frame, compute the number of positive events and negative events
        pe, ne = self.event_decision()

        # update last frame time
        if self.live:
            self.last_frame_time = current_time

        return PoseEventInterface(positive_event=pe, negative_event=ne)

    def update_queue(self, po):
        # if no new input, push None into the queue
        # in the queue, each gaze information is a tuple (whether leaning in / leaning out, whether it is newly output information)
        # for example ("leaning", False) means that at this point, a participant is observed leaning in
        # however, it is not a newly received information
        # it is copied from the previous informtion to smooth frame
        if not po.is_new():
            for body in self.queue:
                # use history information to smooth frames
                i = len(self.queue[body]) - 1
                while (
                    i >= 0 and self.time - self.queue[body][i][1] <= self.smooth_frame
                ):
                    if self.queue[body][i][0][1] is True:
                        self.queue[body].append(
                            ((self.queue[body][i][0][0], False), self.time)
                        )
                        break
                    i -= 1
                else:
                    self.queue[body].append(((None, False), self.time))
        else:
            # if there is new information, push it into the queue
            for body, pose in po.poses:
                if body not in self.positive_cooling_time:
                    self.positive_cooling_time[body] = self.positive_event_time
                    self.negative_cooling_time[body] = self.negative_event_time
                if body in self.queue:
                    self.queue[body].append(((pose, True), self.time))
                else:
                    self.queue[body] = [((pose, True), self.time)]

            # if someone is not in the received feature, treat it as none information and do the same operation
            for body in self.queue:
                if body not in [i for i, _ in po.poses]:
                    i = len(self.queue[body]) - 1
                    while (
                        i >= 0
                        and self.time - self.queue[body][i][1] <= self.smooth_frame
                    ):
                        if self.queue[body][i][0][1] is True:
                            self.queue[body].append(
                                ((self.queue[body][i][0][0], False), self.time)
                            )
                            break
                        i -= 1
                    else:
                        self.queue[body].append(((None, False), self.time))

    def event_decision(self):
        """
        This function decides how many positive events and negatives to report
        """

        positive_event = 0
        negative_event = 0

        file = open(f"./event_log.csv", "a", encoding="utf-8")
        for body in self.queue:
            # find the beginning index of the period which is longer than the leaning out threshold
            begin = 0
            if body == "P1":
                continue
            while begin < len(self.queue[body]) - 1:
                if (
                    self.queue[body][-1][1] - self.queue[body][begin][1]
                    >= self.leanout_count_time
                    and self.queue[body][-1][1] - self.queue[body][begin + 1][1]
                    < self.leanout_count_time
                ):
                    break
                begin += 1

            if begin < len(self.queue[body]) - 1:
                leanout = all(
                    i[0][0] != "leaning in" for i in self.queue[body][begin + 1 :]
                )

                # for a participant, if the negative cooling time is less than 0, check negative posture events
                # use 0.00001 here to mitigate the influence of accuracy of decimal
                if self.negative_cooling_time[body] <= 0.00001:
                    if leanout:
                        negative_event += 1
                        file.write(
                            str(self.time)
                            + ","
                            + str(body)
                            + " negative posture event\n"
                        )
                        # reset all cooling time if a negative event happens
                        self.negative_cooling_time[body] = self.negative_event_time
                        self.positive_cooling_time[body] = self.positive_event_time

                if self.positive_cooling_time[body] <= 0.00001:
                    if not leanout:
                        positive_event += 1
                        file.write(
                            str(self.time)
                            + ","
                            + str(body)
                            + " positive posture event\n"
                        )
                        # by definition, only reset the positive event cooling time
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
