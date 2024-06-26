#include "utils.hpp"
#include <iostream>

void print_calibration_info(k4a_calibration_t calibration) {
  std::cout
      << "Rotation "
      << calibration.color_camera_calibration.extrinsics.rotation[0] << ","
      << calibration.color_camera_calibration.extrinsics.rotation[1] << ","
      << calibration.color_camera_calibration.extrinsics.rotation[2] << ","
      << calibration.color_camera_calibration.extrinsics.rotation[3] << ","
      << calibration.color_camera_calibration.extrinsics.rotation[4] << ","
      << calibration.color_camera_calibration.extrinsics.rotation[5] << ","
      << calibration.color_camera_calibration.extrinsics.rotation[6] << ","
      << calibration.color_camera_calibration.extrinsics.rotation[7] << ","
      << calibration.color_camera_calibration.extrinsics.rotation[8]
      << std::endl;

  std::cout << "Translation "
            << calibration.color_camera_calibration.extrinsics.translation[0]
            << ","
            << calibration.color_camera_calibration.extrinsics.translation[1]
            << ","
            << calibration.color_camera_calibration.extrinsics.translation[2]
            << std::endl;

  std::cout
      << "Camera Matrix "
      << calibration.color_camera_calibration.intrinsics.parameters.param.fx
      << ","
      << calibration.color_camera_calibration.intrinsics.parameters.param.fy
      << ","
      << calibration.color_camera_calibration.intrinsics.parameters.param.cx
      << ","
      << calibration.color_camera_calibration.intrinsics.parameters.param.cy
      << std::endl;

  std::cout
      << "Distortion Coefficient "
      << calibration.color_camera_calibration.intrinsics.parameters.param.k1
      << ","
      << calibration.color_camera_calibration.intrinsics.parameters.param.k2
      << ","
      << calibration.color_camera_calibration.intrinsics.parameters.param.p1
      << ","
      << calibration.color_camera_calibration.intrinsics.parameters.param.p2
      << ","
      << calibration.color_camera_calibration.intrinsics.parameters.param.k3
      << ","
      << calibration.color_camera_calibration.intrinsics.parameters.param.k4
      << ","
      << calibration.color_camera_calibration.intrinsics.parameters.param.k5
      << ","
      << calibration.color_camera_calibration.intrinsics.parameters.param.k6
      << std::endl;
}
