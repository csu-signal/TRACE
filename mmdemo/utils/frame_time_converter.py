class FrameTimeConverter:
    """
    A helper class that can quickly look up what time a frame was processed
    or which frame was being processed at a given time.
    """

    def __init__(self) -> None:
        self.data = []

    def add_data(self, frame, time):
        """
        Add a new datapoint. The frame and time must be strictly increasing
        so binary search can be used.

        Arguments:
        frame -- the frame number
        time -- the current time
        """
        # must be strictly monotonic so binary search can be used
        assert len(self.data) == 0 or frame > self.data[-1][0]
        assert len(self.data) == 0 or time > self.data[-1][1]
        self.data.append((frame, time))

    def get_time(self, frame):
        """
        Return the time that a frame was processed
        """
        return self._binary_search(0, frame)[1]

    def get_frame(self, time):
        """
        Return the frame being processed at a certain time
        """
        return self._binary_search(1, time)[0]

    def get_num_datapoints(self):
        """
        Returns how many datapoints have been added
        """
        return len(self.data)

    def _binary_search(self, index, val):
        assert len(self.data) > 0
        if self.data[-1][index] < val:
            return self.data[-1]
        left = 0
        right = len(self.data)
        while right - left > 1:
            middle = (left + right) // 2
            if self.data[middle][index] < val:
                left = middle
            elif self.data[middle][index] > val:
                right = middle
            else:
                left = middle
                right = middle
        return self.data[left]
