#pragma once

#include <k4abt.hpp>
#include <nlohmann/json.hpp>
#include <pybind11/numpy.h>

using namespace nlohmann;
namespace py = pybind11;

// note: template functions can't be defined in .cpp as easily so
// they are defined here instead

/* Extract predictions of joint positions and orientations from a body
 * tracking frame. Store as json in the expected format for the python
 * library.*/
json body_frame_info(k4abt::frame frame);


/* Turn a json object into a python dictionary */ 
py::dict json_to_dict(json data);

/* Make a 2d numpy array from an array of doubles */
template <size_t size> py::array_t<double> make_1d_array(double arr[size]) {
  py::array_t<double> out(size);
  double *ptr = (double *)out.request().ptr;

  for (size_t i = 0; i < size; i++) {
    ptr[i] = arr[i];
  }

  return out;
}

/* make a 1d numpy array from an array of doubles */
template <size_t rows, size_t cols>
py::array_t<double> make_2d_array(double arr[rows][cols]) {
  py::array_t<double> out(rows * cols);
  double *ptr = (double *)out.request().ptr;

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      ptr[i * rows + j] = arr[i][j];
    }
  }

  return out.reshape({rows, cols});
}
