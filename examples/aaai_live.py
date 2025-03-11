import yaml
from mmdemo_azure_kinect import DeviceType, create_azure_kinect_features

from mmdemo.demo import Demo
from mmdemo.features import (
    AAAIFrame,
    DisplayFrame,
    EngagementLevel,
    GazeBodyTracking,
    GazeEvent,
    GazeSelection,
    Gesture,
    Object,
    Pose,
    PoseEvent,
    SaveVideo,
    SelectedObjects,
)

if __name__ == "__main__":
    with open(
        os.path.join(os.path.dirname(__file__), f"config.yaml"), "r", encoding="utf-8"
    ) as file:
        config = yaml.safe_load(file)

    color, depth, body_tracking, calibration = (
        create_azure_kinect_features(
            DeviceType.PLAYBACK,
            mkv_path=config["mkv_file_path"],
            mkv_frame_rate=config["mkv_frame_rate"],
            playback_frame_rate=config["playback_frame_rate"],
            playback_end_seconds=config["playback_time"],
        )
        if not config["running_alive"]
        else create_azure_kinect_features(DeviceType.CAMERA, camera_index=0)
    )

    gaze = GazeBodyTracking(
        body_tracking,
        calibration,
        left_position=config["left_position"],
        middle_position=config["middle_position"],
    )
    pose = Pose(
        body_tracking,
        left_position=config["left_position"],
        middle_position=config["middle_position"],
    )

    gazeselection = GazeSelection(
        body_tracking,
        calibration,
        gaze,
        left_position=config["left_position"],
        middle_position=config["middle_position"],
    )

    gazeevent = GazeEvent(
        gazeselection,
        config["running_alive"],
        config["playback_frame_rate"],
        config["gaze_window"],
        config["individual_gaze_event"],
        config["group_gaze_event"],
        config["gaze_beginning_buffer"],
        config["gaze_lookaway_buffer"],
        config["smooth_frame"],
        config["speaker"],
    )
    poseevent = PoseEvent(
        pose,
        config["running_alive"],
        config["playback_frame_rate"],
        config["pose_window"],
        config["pose_positive_event"],
        config["pose_negative_event"],
        config["leanout_time"],
        config["smooth_frame"],
    )

    gesture = Gesture(color, depth, body_tracking, calibration)
    # which objects are selected by gesture
    objects = Object(color, depth, calibration)
    selected_objects = SelectedObjects(objects, gesture)  # pyright: ignore

    engagmentlevel = EngagementLevel(
        poseevent,
        gazeevent,
        config["running_alive"],
        config["update_check_interval"],
        config["playback_frame_rate"],
        config["gaze_positive_count_time"],
        config["gaze_negative_count_time"],
        config["posture_positive_count_time"],
        config["posture_negative_count_time"],
    )
    aaaioutputframe = AAAIFrame(
        color=color,
        gaze=gaze,
        gesture=gesture,
        sel_objects=selected_objects,
        calibration=calibration,
        engagement_level=engagmentlevel,
        draw_cone=config["draw_gaze_cone"],
    )

    # run demo and show output
    demo = Demo(
        targets=[
            DisplayFrame(aaaioutputframe),
            # SaveVideo(aaaioutputframe, frame_rate = config["playback_frame_rate"])
        ]
    )

    # demo.show_dependency_graph()
    demo.run()
