from mmdemo.demo import Demo
from mmdemo.features import (
    DepthAnythingV2Metric,
    DisplayFrame,
    MetricDepthVisualization,
    WebcamDevice,
)

if __name__ == "__main__":
    color = WebcamDevice(camera_index=0)

    depth = DepthAnythingV2Metric(color)

    depth_visualization = MetricDepthVisualization(depth)

    demo = Demo(targets=[DisplayFrame(color), DisplayFrame(depth_visualization)])
    demo.run()
