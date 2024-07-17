class IFeature:
    def __init__(self, *args, **kwargs):
        pass

    def processFrame(self, *args, **kwargs):
        raise NotImplementedError
