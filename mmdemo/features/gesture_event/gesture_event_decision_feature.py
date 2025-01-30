from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    GestureConesInterface,
    GestureEventInterface
)

@final
class GestureEvent(BaseFeature[GestureEventInterface]):
    """
    This feature is used to decide whether there is pointing event
    Pointing is treated as a positive signal of engagement
    Queue is also used to store history
    """
    def __init__(self,
        gs: BaseFeature[GestureConesInterface],
        ti: int,
        window: int,
        fr: int
    ) -> None:
        super().__init__(gs)
        self.ti = ti
        self.window = window
        self.fr = fr
    
    def initialize(
        self
    ) -> None:
        #Don't care about exactly who is pointing, only use list [] here
        self.queue = []
        self.count_frame = 0
    
    def get_output(
        self,
        gs: GestureConesInterface
    ) -> GestureEventInterface | None:
        #Count a new frame
        self.count_frame += 1

        #If the queue is full, remove the first historical item
        if len(self.queue) >= self.window * self.fr:
            self.queue.pop(0)

        #no new information received, push None into the queue
        if not gs.is_new():
            self.queue.append(None)
        else:
            #If receive new information
            #Check if someone is pointing
            if gs.handedness == []:
                self.queue.append(None)
            else:
                self.queue.append(True)
        
        #if not the check point time, return None
        if self.count_frame % (self.ti * self.fr) != 0:
            return None
        
        pe = 0
        
        #If someone is pointing in the time window
        #We treat it as a positive event
        for i in self.queue:
            if i is True:
                pe += 1

        return GestureEventInterface(positive_event = pe)