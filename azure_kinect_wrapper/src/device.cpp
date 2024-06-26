#include "device.hpp"
#include "k4abttypes.h"
#include "k4arecord/playback.h"
#include "utils.hpp"
#include <iostream>
#include <nlohmann/json.hpp>

#include "body_tracking_helpers.hpp"

/*
   Device template implementation
*/

using namespace nlohmann;

void Device::open() {
  open_device();

  // TODO: is any of this needed?
  // int depthWidth = calibration.depth_camera_calibration.resolution_width;
  // int depthHeight = calibration.depth_camera_calibration.resolution_height;
  // calibration.color_resolution = K4A_COLOR_RESOLUTION_1080P;
  // k4a_transformation_t transformation = k4a_transformation_create(&calibration);
  // k4a_transformation_destroy(transformation);

  print_calibration_info(calibration);

  // TODO: this only works if cuda is installed
  k4abt_tracker_configuration_t body_tracker = K4ABT_TRACKER_CONFIG_DEFAULT;
  body_tracker.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA;

  k4abt_tracker_t tracker_handle = nullptr;
  K4A_VERIFY(
      k4abt_tracker_create(&calibration, body_tracker, &tracker_handle),
      "Body tracker initialization failed")

  json json_output;
  json_output["k4abt_sdk_version"] = K4ABT_VERSION_STR;

  json_output["camera_calibration"] = json::object();

  // Store all rotation information to the json
  json_output["camera_calibration"]["rotation"] = json::array();
  for (int i = 0; i < 9; i++) {
    json_output["camera_calibration"]["rotation"].push_back(
        calibration.color_camera_calibration.extrinsics.rotation[i]);
  }

  // Store all translation information to the json
  json_output["camera_calibration"]["translation"] = json::array();
  for (int i = 0; i < 3; i++) {
    json_output["camera_calibration"]["translation"].push_back(
        calibration.color_camera_calibration.extrinsics.translation[i]);
  }

  json_output["camera_calibration"]["fx"] =
      calibration.color_camera_calibration.intrinsics.parameters.param.fx;
  json_output["camera_calibration"]["fy"] =
      calibration.color_camera_calibration.intrinsics.parameters.param.fy;
  json_output["camera_calibration"]["cx"] =
      calibration.color_camera_calibration.intrinsics.parameters.param.cx;
  json_output["camera_calibration"]["cy"] =
      calibration.color_camera_calibration.intrinsics.parameters.param.cy;
  json_output["camera_calibration"]["p1"] =
      calibration.color_camera_calibration.intrinsics.parameters.param.p1;
  json_output["camera_calibration"]["p2"] =
      calibration.color_camera_calibration.intrinsics.parameters.param.p2;
  json_output["camera_calibration"]["k1"] =
      calibration.color_camera_calibration.intrinsics.parameters.param.k1;
  json_output["camera_calibration"]["k2"] =
      calibration.color_camera_calibration.intrinsics.parameters.param.k2;
  json_output["camera_calibration"]["k3"] =
      calibration.color_camera_calibration.intrinsics.parameters.param.k3;
  json_output["camera_calibration"]["k4"] =
      calibration.color_camera_calibration.intrinsics.parameters.param.k4;
  json_output["camera_calibration"]["k5"] =
      calibration.color_camera_calibration.intrinsics.parameters.param.k5;
  json_output["camera_calibration"]["k6"] =
      calibration.color_camera_calibration.intrinsics.parameters.param.k6;

  // Store all joint names to the json
  json_output["joint_names"] = json::array();
  for (int i = 0; i < (int)K4ABT_JOINT_COUNT; i++) {
    json_output["joint_names"].push_back(
        g_jointNames.find((k4abt_joint_id_t)i)->second);
  }

  // Store all bone linkings to the json
  json_output["bone_list"] = json::array();
  for (int i = 0; i < (int)g_boneList.size(); i++) {
    json_output["bone_list"].push_back(
        {g_jointNames.find(g_boneList[i].first)->second,
         g_jointNames.find(g_boneList[i].second)->second});
  }

  std::cout << "Device opened successfully" << std::endl;
}

void Device::close() {
  close_device();
  std::cout << "Device closed successfully" << std::endl;
}

/*
   Playback implementation
*/

Playback::Playback(const char *recording_path) {
  path = recording_path;
  open();
}

void Playback::open_device() {
  K4A_VERIFY(k4a_playback_open(path, &playback_handle), "Cannot open recording")

  K4A_VERIFY(k4a_playback_get_calibration(playback_handle, &calibration),
             "Failed to get calibration")

  k4a_playback_set_color_conversion(playback_handle,
                                    K4A_IMAGE_FORMAT_COLOR_BGRA32);
}

void Playback::close_device() {
  k4a_playback_close(playback_handle);
}

void Playback::get_capture() {}

/*
   Camera implementation: TODO
*/

void Camera::open_device() {}

void Camera::close_device() {}

void Camera::get_capture() {}
