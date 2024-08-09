#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <nlohmann/json.hpp>

#include "device.hpp"

namespace py = pybind11;

// define python interface
PYBIND11_MODULE(_azure_kinect, m) {

  py::class_<Device>(m, "Device")
      .def("close", &Device::close)
      .def("get_frame", &Device::get_frame)
      .def("get_calibration_matrices", &Device::get_calibration_matrices)
      .def("get_frame_count", &Device::get_frame_count);

  py::class_<Playback, Device>(m, "Playback")
      .def(py::init<const char *>())
      .def("skip_frames", &Playback::skip_frames);

  py::class_<Camera, Device>(m, "Camera").def(py::init<uint32_t>());
}
