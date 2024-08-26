#pragma once

#include <k4a/k4a.hpp>
#include <k4abt.hpp>
#include <k4arecord/playback.hpp>
#include <pybind11/numpy.h>

namespace py = pybind11;

/* The base class for devices */
struct Device {
  /* Open the device */
  void open();

  /* Close the device */
  void close();

  /* Get frame data (color, depth, body tracking).
   * On failure, return (None, None {})
   */
  py::object get_frame();

  /* Get calibration matrices of device */
  py::object get_calibration_matrices();

  /* Get the current frame count */
  int get_frame_count();

protected:
  /* Abstract method for opening the specific type
   * of device and getting camera calibration
   * information
   */
  virtual void open_device() = 0;

  /* Abstract method for closing the specific type
   * of device
   */
  virtual void close_device() = 0;

  /* Abstract method for updating the capture handing
   * using the specific type of device
   */
  virtual void update_capture_handle() = 0;

  size_t frame_count;

  // stores calibration and capture data
  k4a::calibration calibration;
  k4a::transformation calibration_transform;
  k4a::capture capture_handle;

  // stores body tracker
  k4abt::tracker body_tracker;
};


/* Device for reading mkv files */
struct Playback : public Device {
  /* Constructor
   * recording_path -- path to the mkv file
   */
  Playback(const char *recording_path);

  /* Quickly skip frames by updating the capture handle
   * `n_frames` times without running the body tracker.
   */
  void skip_frames(int n_frames);

protected:
  // implement abstract methods of Device
  void open_device() override;
  void close_device() override;
  void update_capture_handle() override;

private:
  // store playback handle and path
  k4a::playback playback_handle;
  const char *path;
};

/* Device for using live Azure Kinect camera */
struct Camera : public Device {
  /* Constructor
   * camera_index -- the azure kinect camera index
   */
  Camera(uint32_t camera_index);

protected:
  // implement abstract methods of Device
  void open_device() override;
  void close_device() override;
  void update_capture_handle() override;

private:
  // store camera index and handle
  uint32_t camera_index;
  k4a::device camera_handle;
};
