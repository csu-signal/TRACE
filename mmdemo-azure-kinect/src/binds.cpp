#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <nlohmann/json.hpp>

#include "device.hpp"

namespace py = pybind11;

// define python interface
PYBIND11_MODULE(_azure_kinect, m) {

  // Device is the base class for devices
  // and has shared functions
  py::class_<Device>(m, "Device")
      .def("close", &Device::close)
      .def("get_frame", &Device::get_frame)
      .def("get_calibration_matrices", &Device::get_calibration_matrices)
      .def("get_frame_count", &Device::get_frame_count);

  // Playback is a device for reading mkv files
  py::class_<Playback, Device>(m, "Playback")
      .def(py::init<const char *>())
      .def("skip_frames", &Playback::skip_frames);

  // Camera is a device for using live cameras
  py::class_<Camera, Device>(m, "Camera").def(py::init<uint32_t>());
}
