#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <nlohmann/json.hpp>

#include "device.hpp"

namespace py = pybind11;

PYBIND11_MODULE(azure_kinect, m) {

  py::class_<Device>(m, "Device")
      .def("close", &Device::close)
      .def("get_frame", &Device::get_frame)
      .def("get_calibration_matrices", &Device::get_calibration_matrices);

  py::class_<Playback, Device>(m, "Playback").def(py::init<const char *>());

  py::class_<Camera, Device>(m, "Camera").def(py::init<>());
}
