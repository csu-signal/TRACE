from mmdemo.demo import Demo
from mmdemo.features import (
    DepthAnythingV2Metric,
    DepthAnythingV2Relative,
    DisplayFrame,
    MetricDepthVisualization,
    RelativeDepthVisualization,
    WebcamDevice,
)

if __name__ == "__main__":
    color = WebcamDevice(camera_index=0)

    metric_depth = DepthAnythingV2Metric(color)
    relative_depth = DepthAnythingV2Relative(color)

    metric_depth_visualization = MetricDepthVisualization(metric_depth)
    relative_depth_visualization = RelativeDepthVisualization(relative_depth)

    demo = Demo(
        targets=[
            DisplayFrame(color),
            DisplayFrame(metric_depth_visualization, window_name="Metric Depth"),
            DisplayFrame(relative_depth_visualization, window_name="Relative Depth"),
        ]
    )
    demo.run()
