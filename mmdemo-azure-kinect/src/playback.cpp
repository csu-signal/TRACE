#include "device.hpp"
#include <iostream>

// playback implementation

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

void Playback::skip_frames(int n_frames){
  for (int i = 0; i < n_frames; i++) {
    capture_handle.reset();
    update_capture_handle();
    frame_count++;
  }
}
