#pragma once

#include <k4a/k4a.hpp>
#include <k4abt.hpp>
#include <k4arecord/playback.hpp>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct Device {
  void open();
  void close();
  py::object get_frame();
  py::object get_calibration_matrices();

protected:
  k4a::calibration calibration;
  k4a::capture capture_handle;
  // open device and set calibration
  virtual void open_device() = 0;
  // close device
  virtual void close_device() = 0;
  // set capture handle to newest/most recent capture
  virtual void update_capture_handle() = 0;

  size_t frame_count;
  k4a::transformation calibration_transform;
  k4abt::tracker body_tracker;
};

struct Playback : public Device {
  Playback(const char *recording_path);
  void skip_frames(int n_frames);

protected:
  void open_device() override;
  void close_device() override;
  void update_capture_handle() override;

private:
  k4a::playback playback_handle;
  const char *path;
};

struct Camera : public Device {
  Camera(uint32_t camera_index);

protected:
  void open_device() override;
  void close_device() override;
  void update_capture_handle() override;

private:
  uint32_t camera_index;
  k4a::device camera_handle;
};
