#include "device.hpp"
#include <iostream>

// camera implementation

Camera::Camera(uint32_t camera_index) {
  this->camera_index = camera_index;
  std::cout << "Camera Index: "
            << camera_index << std::endl;
  open();
}

void Camera::open_device() {
  std::cout << "Attempting to open camera with index: "
            << this->camera_index << std::endl;
  camera_handle = k4a::device::open(this->camera_index);

  k4a_device_configuration_t config = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
  config.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
  config.color_resolution = K4A_COLOR_RESOLUTION_1080P;
  config.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;

  camera_handle.start_cameras(&config);

  calibration =
      camera_handle.get_calibration(config.depth_mode, config.color_resolution);

  std::cout << "opening camera with serial number: "
            << camera_handle.get_serialnum() << std::endl;
}

void Camera::close_device() {
  camera_handle.stop_cameras();
  camera_handle.close();
}

void Camera::update_capture_handle() {
  camera_handle.get_capture(&capture_handle);
}
