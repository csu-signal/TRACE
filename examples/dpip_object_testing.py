from mmdemo.demo import Demo
from mmdemo.features import (
    DisplayFrame,
    DpipBlockDetectionsFrame,
    DpipObject,
    DpipObjectsFrame,
    WebcamDevice,
)

if __name__ == "__main__":
    color = WebcamDevice(camera_index=0)

    objects = DpipObject(color)

    block_detections_frame = DpipBlockDetectionsFrame(objects)
    objects_frame = DpipObjectsFrame(color, objects)

    demo = Demo(
        targets=[
            DisplayFrame(objects_frame),
            DisplayFrame(block_detections_frame),
        ]
    )
    demo.run()
    demo.print_time_benchmarks()
