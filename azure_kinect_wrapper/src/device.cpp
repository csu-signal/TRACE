#include "device.hpp"
#include "k4arecord/playback.h"
#include "utils.hpp"
#include <iostream>

/*
   Device template implementation
*/

void Device::open() {
  open_device();
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
