#include "device.hpp"
#include "utils.hpp"
#include <iostream>
#include <nlohmann/json.hpp>
#include <pybind11/numpy.h>

/*
   Device template implementation
*/

using namespace nlohmann;
namespace py = pybind11;

void Device::open() {
  open_device();

  calibration_transform = k4a::transformation(calibration);
  frame_count = 0;
  std::cout << "Device opened" << std::endl;

  // TODO: this only works if cuda is availible
  // I was getting weird errors with the default processing mode
  k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
  tracker_config.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA;

  body_tracker = k4abt::tracker::create(calibration, tracker_config);
  std::cout << "Body tracker initialized" << std::endl;
}

py::object Device::get_calibration_matrices() {
  k4a_calibration_intrinsic_parameters_t ip =
      calibration.color_camera_calibration.intrinsics.parameters;
  k4a_calibration_extrinsics_t e =
      calibration.color_camera_calibration.extrinsics;

  double camera_matrix_arr[3][3] = {
      {ip.param.fx, 0, ip.param.cx}, {0, ip.param.fy, ip.param.cy}, {0, 0, 1}};
  py::array_t<double> camera_matrix = make_2d_array<3, 3>(camera_matrix_arr);

  double rotation_arr[3][3] = {{e.rotation[0], e.rotation[1], e.rotation[2]},
                               {e.rotation[3], e.rotation[4], e.rotation[5]},
                               {e.rotation[6], e.rotation[7], e.rotation[8]}};
  py::array_t<double> rotation = make_2d_array<3, 3>(rotation_arr);

  double translation_arr[3] = {e.translation[0], e.translation[1],
                               e.translation[2]};
  py::array_t<double> translation = make_1d_array<3>(translation_arr);

  double distortion_arr[8] = {
      ip.param.k1, ip.param.k2, ip.param.p1, ip.param.p2,
      ip.param.k3, ip.param.k4, ip.param.k5, ip.param.k6,
  };
  py::array_t<double> distortion = make_1d_array<8>(distortion_arr);

  return py::make_tuple(camera_matrix, rotation, translation, distortion);
}

void Device::close() {
  body_tracker.shutdown();
  std::cout << "Body tracker shutdown" << std::endl;

  close_device();
  std::cout << "Device closed successfully" << std::endl;
}

py::dict json_to_dict(json data) {
  return py::module::import("json").attr("loads")(data.dump());
}

py::object get_frame_fail_return() {
  return py::make_tuple(py::none(), py::none(), json_to_dict(json::object()));
}

py::object Device::get_frame() {
  capture_handle.reset();
  update_capture_handle();
  frame_count++;

  if (!capture_handle.is_valid()) {
    return get_frame_fail_return();
  }

  k4a::image depth_image = capture_handle.get_depth_image();
  k4a::image color_image = capture_handle.get_color_image();
  if (!depth_image.is_valid() || !color_image.is_valid()) {
    return get_frame_fail_return();
  }

  // enqueue to body tracker so it has as much time as possible to process
  if (!body_tracker.enqueue_capture(capture_handle)) {
    return get_frame_fail_return();
  };

  // create transformed depth image array
  k4a::image transformed_depth_image =
      calibration_transform.depth_image_to_color_camera(depth_image);
  py::array_t<uint16_t> transformed_depth_array(
      {(size_t)transformed_depth_image.get_height_pixels(),
       (size_t)transformed_depth_image.get_width_pixels()},
      (uint16_t *)transformed_depth_image.get_buffer());

  // create color image array
  py::array_t<uint8_t> color_array({(size_t)color_image.get_height_pixels(),
                                    (size_t)color_image.get_width_pixels(),
                                    (size_t)4},
                                   color_image.get_buffer());

  // get body tracking info (enqueued above)
  k4abt::frame body_frame = body_tracker.pop_result();
  if (body_frame == nullptr) {
    return get_frame_fail_return();
  }
  json body_frame_info_json = body_frame_info(body_frame);

  return py::make_tuple(color_array, transformed_depth_array,
                        json_to_dict(body_frame_info_json));
}

/*
   Playback implementation
*/

Playback::Playback(const char *recording_path) {
  path = recording_path;
  open();
}

void Playback::open_device() {
  playback_handle = k4a::playback::open(path);
  playback_handle.set_color_conversion(K4A_IMAGE_FORMAT_COLOR_BGRA32);
  calibration = playback_handle.get_calibration();
}

void Playback::close_device() { playback_handle.close(); }

void Playback::update_capture_handle() {
  bool success = playback_handle.get_next_capture(&capture_handle);
  if (!success) {
    std::cout << "no more frames" << std::endl;
  }
}

/*
   Camera implementation: TODO
*/

Camera::Camera() {
  std::cout << "camera not implemented" << std::endl;
  assert(false);
}

void Camera::open_device() {
  std::cout << "camera not implemented" << std::endl;
  assert(false);
}

void Camera::close_device() {
  std::cout << "camera not implemented" << std::endl;
  assert(false);
}

void Camera::update_capture_handle() {
  std::cout << "camera not implemented" << std::endl;
  assert(false);
}
