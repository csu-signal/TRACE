#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <k4a/k4a.h>
#include <k4abt.h>
#include <k4arecord/playback.h>

struct Device {
  void open();
  void close();
  virtual void get_capture() = 0;
protected:
  k4a_calibration_t calibration;
  // open device and set calibration
  virtual void open_device() = 0;
  // close device
  virtual void close_device() = 0;
};

struct Playback : public Device {
  Playback(const char *recording_path);
  void get_capture() override;
protected:
  void open_device() override;
  void close_device() override;
  k4a_playback_t playback_handle;
  const char *path;
};

struct Camera : public Device {
  void get_capture() override;
protected:
  void open_device() override;
  void close_device() override;
};

#endif
