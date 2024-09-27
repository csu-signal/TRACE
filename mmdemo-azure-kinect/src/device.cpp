#include "device.hpp"
#include "utils.hpp"
#include <iostream>
#include <k4a/k4a.hpp>
#include <nlohmann/json.hpp>
#include <pybind11/numpy.h>

using namespace nlohmann;
namespace py = pybind11;

void Device::open() {
  // TODO: if this fails, an error will be thrown from C++, which causes
  // pytest to crash for some reason (this is why some azure kinect tests
  // are skipped). This could be rewritten to set a variable that could be
  // checked from python instead of erroring, but would probably be annoying
  // to do because it would have to use the C SDK function and then create a 
  // C++ class using the result of that.
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


void Device::close() {
  body_tracker.shutdown();
  std::cout << "Body tracker shutdown" << std::endl;

  close_device();
  std::cout << "Device closed successfully" << std::endl;
}

py::object Device::get_calibration_matrices() {
  k4a_calibration_intrinsic_parameters_t ip =
      calibration.color_camera_calibration.intrinsics.parameters;
  k4a_calibration_extrinsics_t e =
      calibration.color_camera_calibration.extrinsics;

  // extract required calibration matrices from camera calibration params
  double camera_matrix_arr[3][3] = {
      {ip.param.fx, 0, ip.param.cx}, {0, ip.param.fy, ip.param.cy}, {0, 0, 1}};

  double rotation_arr[3][3] = {{e.rotation[0], e.rotation[1], e.rotation[2]},
                               {e.rotation[3], e.rotation[4], e.rotation[5]},
                               {e.rotation[6], e.rotation[7], e.rotation[8]}};

  double translation_arr[3] = {e.translation[0], e.translation[1],
                               e.translation[2]};

  double distortion_arr[8] = {
      ip.param.k1, ip.param.k2, ip.param.p1, ip.param.p2,
      ip.param.k3, ip.param.k4, ip.param.k5, ip.param.k6,
  };

  // create numpy arrays and return a tuple
  return py::make_tuple(
      make_2d_array<3, 3>(camera_matrix_arr),
      make_2d_array<3, 3>(rotation_arr),
      make_1d_array<3>(translation_arr),
      make_1d_array<8>(distortion_arr)
  );
}

int Device::get_frame_count(){
    return frame_count;
}

/* Create the object that is returned on failure of `Device::get_frame()` */
py::object get_frame_fail_return() {
  return py::make_tuple(py::none(), py::none(), json_to_dict(json::object()));
}

py::object Device::get_frame() {
  // get next capture and increment frame count
  capture_handle.reset();
  update_capture_handle();
  frame_count++;

  if (!capture_handle.is_valid()) {
    return get_frame_fail_return();
  }

  // get depth and color image from capture
  k4a::image depth_image = capture_handle.get_depth_image();
  k4a::image color_image = capture_handle.get_color_image();
  if (!depth_image.is_valid() || !color_image.is_valid()) {
    return get_frame_fail_return();
  }

  // enqueue capture to body tracker so it has as much time as
  // possible to process
  if (!body_tracker.enqueue_capture(capture_handle)) {
    return get_frame_fail_return();
  };

  // Create a new depth image that has been transformed such that
  // its pixels align with the color image.
  k4a::image transformed_depth_image =
      calibration_transform.depth_image_to_color_camera(depth_image);

  // create a numpy array from the transformed depth image
  py::array_t<uint16_t> transformed_depth_array(
      {(size_t)transformed_depth_image.get_height_pixels(),
       (size_t)transformed_depth_image.get_width_pixels()},
      (uint16_t *)transformed_depth_image.get_buffer());

  // create numpy array from color image
  py::array_t<uint8_t> color_array({(size_t)color_image.get_height_pixels(),
                                    (size_t)color_image.get_width_pixels(),
                                    (size_t)4},
                                   color_image.get_buffer());

  // wait for body tracking info (enqueued above)
  k4abt::frame body_frame = body_tracker.pop_result();
  if (body_frame == nullptr) {
    return get_frame_fail_return();
  }

  // convert body frame to json object with data we care about
  json body_frame_info_json = body_frame_info(body_frame);

  return py::make_tuple(color_array, transformed_depth_array,
                        json_to_dict(body_frame_info_json));
}
