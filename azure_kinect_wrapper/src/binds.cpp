#include <pybind11/pybind11.h>

#include <nlohmann/json.hpp>

#include "device.hpp"

namespace py = pybind11;

PYBIND11_MODULE(azure_kinect, m) {

  py::class_<Playback>(m, "Playback")
      .def(py::init<const char *>())
      // open happens during init
      .def("close", &Playback::close);

}
