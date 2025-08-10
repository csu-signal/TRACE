from mmdemo.demo import Demo
from mmdemo.features import (
    DepthAnythingV2Metric,
    DepthAnythingV2Relative,
    DisplayFrame,
    DpipBlockDetectionsFrame,
    DpipObject,
    DpipObjectsFrame,
    MetricDepthVisualization,
    RelativeDepthVisualization,
    WebcamDevice,
)

if __name__ == "__main__":
    color = WebcamDevice(camera_index=0)

    relative_depth = DepthAnythingV2Relative(color)
    # metric_depth = DepthAnythingV2Metric(color)

    relative_depth_frame = RelativeDepthVisualization(relative_depth)
    # metric_depth_frame = MetricDepthVisualization(metric_depth)

    objects = DpipObject(color, relative_depth)
    # objects = DpipObject(color, metric_depth, is_metric_depth=True)

    block_detections_frame = DpipBlockDetectionsFrame(objects)
    objects_frame = DpipObjectsFrame(color, objects)

    demo = Demo(
        targets=[
            DisplayFrame(objects_frame),
            DisplayFrame(block_detections_frame),
            DisplayFrame(relative_depth_frame),
        ]
    )
    demo.run()
    demo.print_time_benchmarks()
