from mmdemo.demo import Demo
from mmdemo.features import DisplayFrame, WebcamDevice

if __name__ == "__main__":
    color = WebcamDevice(camera_index=0)

    demo = Demo(targets=[DisplayFrame(color)])
    demo.run()
