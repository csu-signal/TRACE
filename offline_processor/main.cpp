// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>

#include <k4a/k4a.h>
#include <k4arecord/playback.h>
#include <k4abt.h>
#include <nlohmann/json.hpp>

#include <BodyTrackingHelpers.h>
#include <Utilities.h>
#include <opencv2/opencv.hpp>
#include<Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <conio.h>

#ifdef _DEBUG
#   define Py_DEBUG
#endif

using namespace cv;
using namespace std;
using namespace nlohmann;

PyObject* initalizePython()
{
  PyObject* pInt;
  if (-1 == _putenv("PYTHONHOME=C:\\Users\\vanderh\\Anaconda3\\envs\\handTrackingEnviroment\\")) {
    printf("putenv failed \n");
    return NULL;
  }

  Py_Initialize();
  #ifndef _DEBUG
    import_array();
  #endif
  PyRun_SimpleString("import sys");
  PyRun_SimpleString("import os");
  PyRun_SimpleString("sys.path.append(os.getcwd())");
  PyRun_SimpleString("print('Initalizing Embedded Python')");

  PyObject* myModule = PyImport_ImportModule("pythonCalls");
  return myModule;
}

void finalizePython()
{
  PyRun_SimpleString("print('Finalizing Embedded Python')");
  Py_Finalize();
}

void callOpenFrame(PyObject* pyModule, char* path, int frame_count)
{
  PyObject* myFunction = PyObject_GetAttrString(pyModule, (char*)"openFrame");
  PyObject* args = PyBytes_FromString(path);
  PyObject* myResult = PyObject_CallFunctionObjArgs(myFunction, args, PyFloat_FromDouble(frame_count), NULL);
}

void callOpenFrameBytes(PyObject* pyModule, cv::Mat image, int frame_count)
{
  PyObject* myFunction = PyObject_GetAttrString(pyModule, (char*)"openFrameBytes");

  npy_intp dimensions[3] = { image.rows, image.cols, image.channels() };
  PyObject* pyObject = PyArray_SimpleNewFromData(image.dims + 1, (npy_intp*)&dimensions, NPY_UINT8, image.data);

  PyObject* myResult = PyObject_CallFunctionObjArgs(myFunction, pyObject, PyFloat_FromDouble(frame_count), NULL);
}

bool predict_joints(json& frames_json, int frame_count, k4abt_tracker_t tracker, k4a_capture_t capture_handle)
{
  k4a_wait_result_t queue_capture_result = k4abt_tracker_enqueue_capture(tracker, capture_handle, K4A_WAIT_INFINITE);
  if (queue_capture_result != K4A_WAIT_RESULT_SUCCEEDED)
  {
    cerr << "Error! Adding capture to tracker process queue failed!" << endl;
    return false;
  }

  k4abt_frame_t body_frame = nullptr;
  k4a_wait_result_t pop_frame_result = k4abt_tracker_pop_result(tracker, &body_frame, K4A_WAIT_INFINITE);
  if (pop_frame_result != K4A_WAIT_RESULT_SUCCEEDED)
  {
    cerr << "Error! Popping body tracking result failed!" << endl;
    return false;
  }

  uint32_t num_bodies = k4abt_frame_get_num_bodies(body_frame);
  uint64_t timestamp = k4abt_frame_get_device_timestamp_usec(body_frame);

  json frame_result_json;
  frame_result_json["timestamp_usec"] = timestamp;
  frame_result_json["frame_id"] = frame_count;
  frame_result_json["num_bodies"] = num_bodies;
  frame_result_json["bodies"] = json::array();
  for (uint32_t i = 0; i < num_bodies; i++)
  {
    k4abt_skeleton_t skeleton;
    VERIFY(k4abt_frame_get_body_skeleton(body_frame, i, &skeleton), "Get body from body frame failed!");
    json body_result_json;
    int body_id = k4abt_frame_get_body_id(body_frame, i);
    body_result_json["body_id"] = body_id;

    for (int j = 0; j < (int)K4ABT_JOINT_COUNT; j++)
    {
      body_result_json["joint_positions"].push_back({ skeleton.joints[j].position.xyz.x,
                                                          skeleton.joints[j].position.xyz.y,
                                                          skeleton.joints[j].position.xyz.z });

      body_result_json["joint_orientations"].push_back({ skeleton.joints[j].orientation.wxyz.w,
                                                          skeleton.joints[j].orientation.wxyz.x,
                                                          skeleton.joints[j].orientation.wxyz.y,
                                                          skeleton.joints[j].orientation.wxyz.z });
    }
    frame_result_json["bodies"].push_back(body_result_json);
  }
  frames_json.push_back(frame_result_json);
  k4abt_frame_release(body_frame);

  return true;
}

void check_color_image_exists(PyObject* pyModule, k4a_capture_t capture, k4a_calibration_t calibration, k4a_transformation_t transformation, int frame_count, const char* output_path)
{
  cv::Mat imBGRA;
  k4a_image_t color_image = NULL;
  k4a_image_t ir_image = NULL;

  // Get a color image
  color_image = k4a_capture_get_color_image(capture);

  if (color_image == 0)
  {
    cout << "Failed to get color image from capture" << endl;

  }
  int color_image_width_pixels = k4a_image_get_width_pixels(color_image);
  int color_image_height_pixels = k4a_image_get_height_pixels(color_image);
  imBGRA = cv::Mat(color_image_height_pixels, color_image_width_pixels, CV_8UC4, (void*)k4a_image_get_buffer(color_image));

  #ifndef _DEBUG
    callOpenFrameBytes(pyModule, imBGRA, frame_count);
  #endif

  #ifdef _DEBUG
    char output[1000];
    strcpy_s(output, output_path);
    strcat_s(output, std::to_string(frame_count).c_str());
    strcat_s(output, ".png");

    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(0);

    try {
      //_putenv_s("OPENCV_IO_ENABLE_OPENEXR", "1");
      //_putenv_s("DWITH_JPEG", "1");
      cv::imwrite(output, imBGRA, compression_params);
      //cv::imshow("test", depthMat);
    }
    catch (cv::Exception& e) {
      std::cout << e.msg << std::endl;
    }

    callOpenFrame(pyModule, output, frame_count);
  #endif

  k4a_image_release(color_image);
}

bool check_depth_image_exists(PyObject* pyModule, k4a_capture_t capture, k4a_calibration_t calibration, k4a_transformation_t transformation, int frame_count, const char* output_path)
{
  k4a_image_t transformed_depth_image = NULL;
  k4a_image_t depth = k4a_capture_get_depth_image(capture);
  if (depth != nullptr)
  {
    k4a_image_format_t format = k4a_image_get_format(depth);
    if (K4A_RESULT_SUCCEEDED != k4a_image_create(format,
      calibration.color_camera_calibration.resolution_width,
      calibration.color_camera_calibration.resolution_height,
      0,
      &transformed_depth_image))
    {
      cout << "Failed to create transformed depth image" << endl;
      return false;
    }

    if (K4A_RESULT_SUCCEEDED != k4a_transformation_depth_image_to_color_camera(transformation, depth, transformed_depth_image))
    {
      cout << "Failed to compute transformed depth image" << endl;
      return false;
    }

    if (transformed_depth_image != NULL)
    {
      // you can check the format with this function
      k4a_image_format_t format = k4a_image_get_format(transformed_depth_image);

      // get raw buffer
      uint8_t* buffer = k4a_image_get_buffer(transformed_depth_image);
      uint16_t* depth_buffer = reinterpret_cast<uint16_t*>(buffer);

      // convert the raw buffer to cv::Mat
      int rows = k4a_image_get_height_pixels(transformed_depth_image);
      int cols = k4a_image_get_width_pixels(transformed_depth_image);
      //cv::Mat colorMat(rows, cols, CV_16UC4, (void*)depth_buffer, cv::Mat::AUTO_STEP);
      cv::Mat colorMat(rows, cols, CV_16UC1, (void*)depth_buffer, cv::Mat::AUTO_STEP);

      char output[1000];
      strcpy_s(output, output_path);
      strcat_s(output, std::to_string(frame_count).c_str());
      strcat_s(output, ".png");

      vector<int> compression_params;
      compression_params.push_back(IMWRITE_PNG_COMPRESSION);
      compression_params.push_back(0);

      try {
        //_putenv_s("OPENCV_IO_ENABLE_OPENEXR", "1");
        //_putenv_s("DWITH_JPEG", "1");
        cv::imwrite(output, colorMat, compression_params);
        //cv::imshow("test", depthMat);
      }
      catch (cv::Exception& e) {
        std::cout << e.msg << std::endl;
      }
    }

    k4a_image_release(depth);
    k4a_image_release(transformed_depth_image);
    return true;
  }
  else
  {
    return false;
  }
}

bool process_mkv_offline(PyObject* pyModule, bool camera, const char* input_path, const char* output_path, const char* depth_output_path, const char* rgb_output_path, const char* output_file_name, k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT)
{
  k4a_calibration_t calibration;
  k4a_playback_t playback_handle = nullptr;
  k4a_device_t device = nullptr;
  if (!camera) {
    //File Playback
    k4a_result_t result = k4a_playback_open(input_path, &playback_handle);
    if (result != K4A_RESULT_SUCCEEDED)
    {
      cerr << "Cannot open recording at " << input_path << endl;
      return false;
    }

    //File calibration
    result = k4a_playback_get_calibration(playback_handle, &calibration);
    if (result != K4A_RESULT_SUCCEEDED)
    {
      cerr << "Failed to get calibration" << endl;
      return false;
    }
  }
  else
  {
    //Video Playback
    VERIFY(k4a_device_open(0, &device), "Open K4A Device failed!");

    // Start camera. Make sure depth camera is enabled.
    k4a_device_configuration_t deviceConfig = K4A_DEVICE_CONFIG_INIT_DISABLE_ALL;
    deviceConfig.depth_mode = K4A_DEPTH_MODE_NFOV_UNBINNED;
    deviceConfig.color_resolution = K4A_COLOR_RESOLUTION_1080P;
    deviceConfig.color_format = K4A_IMAGE_FORMAT_COLOR_BGRA32;
    VERIFY(k4a_device_start_cameras(device, &deviceConfig), "Start K4A cameras failed!");
    VERIFY(k4a_device_get_calibration(device, deviceConfig.depth_mode, deviceConfig.color_resolution, &calibration),
      "Get depth camera calibration failed!");
  }

  int depthWidth = calibration.depth_camera_calibration.resolution_width;
  int depthHeight = calibration.depth_camera_calibration.resolution_height;

  //calibration.color_resolution = K4A_COLOR_RESOLUTION_1080P;
  k4a_transformation_t transformation = k4a_transformation_create(&calibration);

  cout << "Rotation " << calibration.color_camera_calibration.extrinsics.rotation[0] <<
    "," <<
    calibration.color_camera_calibration.extrinsics.rotation[1] <<
    "," <<
    calibration.color_camera_calibration.extrinsics.rotation[2] <<
    "," <<
    calibration.color_camera_calibration.extrinsics.rotation[3] <<
    "," <<
    calibration.color_camera_calibration.extrinsics.rotation[4] <<
    "," <<
    calibration.color_camera_calibration.extrinsics.rotation[5] <<
    "," <<
    calibration.color_camera_calibration.extrinsics.rotation[6] <<
    "," <<
    calibration.color_camera_calibration.extrinsics.rotation[7] <<
    "," <<
    calibration.color_camera_calibration.extrinsics.rotation[8] << endl;

  cout << "Translation " << calibration.color_camera_calibration.extrinsics.translation[0] <<
    "," <<
    calibration.color_camera_calibration.extrinsics.translation[1] <<
    "," <<
    calibration.color_camera_calibration.extrinsics.translation[2] << endl;

  cout << "Camera Matrix " << calibration.color_camera_calibration.intrinsics.parameters.param.fx <<
    "," <<
    calibration.color_camera_calibration.intrinsics.parameters.param.fy <<
    "," <<
    calibration.color_camera_calibration.intrinsics.parameters.param.cx <<
    "," <<
    calibration.color_camera_calibration.intrinsics.parameters.param.cy << endl;

  cout << "Distortion Coefficient " << calibration.color_camera_calibration.intrinsics.parameters.param.k1 <<
    "," <<
    calibration.color_camera_calibration.intrinsics.parameters.param.k2 <<
    "," <<
    calibration.color_camera_calibration.intrinsics.parameters.param.p1 <<
    "," <<
    calibration.color_camera_calibration.intrinsics.parameters.param.p2 <<
    "," <<
    calibration.color_camera_calibration.intrinsics.parameters.param.k3 <<
    "," <<
    calibration.color_camera_calibration.intrinsics.parameters.param.k4 <<
    "," <<
    calibration.color_camera_calibration.intrinsics.parameters.param.k5 <<
    "," <<
    calibration.color_camera_calibration.intrinsics.parameters.param.k6 << endl;

  k4abt_tracker_t tracker = NULL;
  if (K4A_RESULT_SUCCEEDED != k4abt_tracker_create(&calibration, tracker_config, &tracker))
  {
    cerr << "Body tracker initialization failed!" << endl;
    return false;
  }

  json json_output;
  json_output["k4abt_sdk_version"] = K4ABT_VERSION_STR;
  if (camera)
  {
    json_output["source_file"] = "Camera";
  }
  else
  {
    json_output["source_file"] = input_path;
  }

  json_output["camera_calibration"] = json::object();

  // Store all rotation information to the json
  json_output["camera_calibration"]["rotation"] = json::array();
  for (int i = 0; i < 9; i++)
  {
    json_output["camera_calibration"]["rotation"].push_back(calibration.color_camera_calibration.extrinsics.rotation[i]);
  }

  // Store all translation information to the json
  json_output["camera_calibration"]["translation"] = json::array();
  for (int i = 0; i < 3; i++)
  {
    json_output["camera_calibration"]["translation"].push_back(calibration.color_camera_calibration.extrinsics.translation[i]);
  }

  json_output["camera_calibration"]["fx"] = calibration.color_camera_calibration.intrinsics.parameters.param.fx;
  json_output["camera_calibration"]["fy"] = calibration.color_camera_calibration.intrinsics.parameters.param.fy;
  json_output["camera_calibration"]["cx"] = calibration.color_camera_calibration.intrinsics.parameters.param.cx;
  json_output["camera_calibration"]["cy"] = calibration.color_camera_calibration.intrinsics.parameters.param.cy;
  json_output["camera_calibration"]["p1"] = calibration.color_camera_calibration.intrinsics.parameters.param.p1;
  json_output["camera_calibration"]["p2"] = calibration.color_camera_calibration.intrinsics.parameters.param.p2;
  json_output["camera_calibration"]["k1"] = calibration.color_camera_calibration.intrinsics.parameters.param.k1;
  json_output["camera_calibration"]["k2"] = calibration.color_camera_calibration.intrinsics.parameters.param.k2;
  json_output["camera_calibration"]["k3"] = calibration.color_camera_calibration.intrinsics.parameters.param.k3;
  json_output["camera_calibration"]["k4"] = calibration.color_camera_calibration.intrinsics.parameters.param.k4;
  json_output["camera_calibration"]["k5"] = calibration.color_camera_calibration.intrinsics.parameters.param.k5;
  json_output["camera_calibration"]["k6"] = calibration.color_camera_calibration.intrinsics.parameters.param.k6;

  // Store all joint names to the json
  json_output["joint_names"] = json::array();
  for (int i = 0; i < (int)K4ABT_JOINT_COUNT; i++)
  {
    json_output["joint_names"].push_back(g_jointNames.find((k4abt_joint_id_t)i)->second);
  }

  // Store all bone linkings to the json
  json_output["bone_list"] = json::array();
  for (int i = 0; i < (int)g_boneList.size(); i++)
  {
    json_output["bone_list"].push_back({ g_jointNames.find(g_boneList[i].first)->second,
                                         g_jointNames.find(g_boneList[i].second)->second });
  }

  cout << "Tracking " << input_path << endl;

  int frame_count = 0;
  json frames_json = json::array();
  bool success = true;
  while (true)
  {
    k4a_capture_t capture_handle = nullptr;
    int stream_result;

    if (!camera)
    {
      //File Based next capture
      stream_result = k4a_playback_get_next_capture(playback_handle, &capture_handle);
    }
    else {
      //Camera Based Next capture
      stream_result = k4a_device_get_capture(device, &capture_handle, 0);
    }

    if (stream_result == K4A_STREAM_RESULT_EOF)
    {
      break;
    }

    cout << "frame " << frame_count << '\r';
    if (stream_result == K4A_STREAM_RESULT_SUCCEEDED)
    {
      // Only try to predict joints when capture contains depth image
      if (check_depth_image_exists(pyModule, capture_handle, calibration, transformation, frame_count, depth_output_path))
      {
        if (camera)
        {
          check_color_image_exists(pyModule, capture_handle, calibration, transformation, frame_count, rgb_output_path);
        }
        success = predict_joints(frames_json, frame_count, tracker, capture_handle);
        k4a_capture_release(capture_handle);
        if (!success)
        {
          cerr << "Predict joints failed for clip at frame " << frame_count << endl;
          break;
        }
      }
    }
    else
    {
      success = false;
      cerr << "Stream error for clip at frame " << frame_count << endl;
      break;
    }

    frame_count++;
  }

  if (success)
  {
    json_output["frames"] = frames_json;
    cout << endl << "DONE " << endl;

    cout << "Total read " << frame_count << " frames" << endl;

    char output[1000];
    strcpy_s(output, output_path);
    strcat_s(output, output_file_name);
    strcat_s(output, ".json");

    std::ofstream output_file(output);
    output_file << std::setw(4) << json_output << std::endl;
    cout << "Results saved in " << output;
  }

  k4abt_tracker_shutdown(tracker);
  k4abt_tracker_destroy(tracker);

  if (!camera)
  {
    //Close File
    k4a_playback_close(playback_handle);
  }
  else
  {
    //Close Camera
    k4a_device_close(device);
  }

  return success;
}

void PrintUsage()
{
#ifdef _WIN32
  cout << "Usage: k4abt_offline_processor.exe <input_mkv_file> <output_json_file> [processing_mode] [-model MODEL_FILE_PATH]\n\t[Optional] processing_mode\n\t\tCPU\n\t\tCUDA\n\t\tTensorRT\n\t\tDirectML ( default )" << endl;
#else
  cout << "Usage: k4abt_offline_processor.exe <input_mkv_file> <output_json_file> [processing_mode] [-model MODEL_FILE_PATH]\n\t[Optional] processing_mode\n\t\tCPU\n\t\tCUDA ( default )\n\t\tTensorRT" << endl;
#endif
}

bool ProcessArguments(k4abt_tracker_configuration_t& tracker_config, int argc, char** argv)
{
  if (argc < 3)
  {
    PrintUsage();
    return false;
  }
  for (int i = 3; i < argc; i++)
  {
    if (0 == strcmp(argv[i], "TensorRT"))
    {
      tracker_config.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU_TENSORRT;
    }
    else if (0 == strcmp(argv[i], "CUDA"))
    {
      tracker_config.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU_CUDA;
    }
    else if (0 == strcmp(argv[i], "CPU"))
    {
      tracker_config.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_CPU;
    }
#ifdef _WIN32
    else if (0 == strcmp(argv[i], "DirectML"))
    {
      tracker_config.processing_mode = K4ABT_TRACKER_PROCESSING_MODE_GPU_DIRECTML;
    }
#endif
    else if (0 == strcmp(argv[i], "-model"))
    {
      if (i < argc - 1)
        tracker_config.model_path = argv[++i];
      else
      {
        printf("Error: model filepath missing\n");
        PrintUsage();
        return false;
      }
    }
    else
    {
      PrintUsage();
      return false;
    }
  }
  return true;
}

int main(int argc, char** argv)
{
  PyObject* pyModule = initalizePython();
  k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
  /*  if (!ProcessArguments(tracker_config, argc, argv))
        return -1;
    return process_mkv_offline(argv[1], argv[2], tracker_config) ? 0 : -1;*/

  return process_mkv_offline(pyModule, true, "F:\\Weights_Task\\Data\\Fib_weights_original_videos\\Group_03-master.mkv", "\\", "Camera1_Depth\\", "Camera1_Rgb\\", "Camera1", tracker_config) ? 0 : -1;
  //callOpenFrame(pyModule, (char*)"Camera1_testData\\0.png");
  //callOpenFrame(pyModule, (char*)"Camera1_testData\\27.png");
  //callOpenFrame(pyModule, (char*)"Camera1_testData\\110.png");
}