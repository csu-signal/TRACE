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

using namespace cv;
using namespace std;
using namespace nlohmann;

bool predict_joints(json &frames_json, int frame_count, k4abt_tracker_t tracker, k4a_capture_t capture_handle)
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
            body_result_json["joint_positions"].push_back( {    skeleton.joints[j].position.xyz.x,
                                                                skeleton.joints[j].position.xyz.y,
                                                                skeleton.joints[j].position.xyz.z });

            body_result_json["joint_orientations"].push_back({  skeleton.joints[j].orientation.wxyz.w,
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

bool check_depth_image_exists(k4a_capture_t capture, const char* output_path, const char* output_file_name)
{
    k4a_image_t depth = k4a_capture_get_depth_image(capture);
    if (depth != nullptr)
    {
        // you can check the format with this function
        k4a_image_format_t format = k4a_image_get_format(depth); // K4A_IMAGE_FORMAT_DEPTH16 

        // get raw buffer
        uint8_t* buffer = k4a_image_get_buffer(depth);

        // convert the raw buffer to cv::Mat
        int rows = k4a_image_get_height_pixels(depth);
        int cols = k4a_image_get_width_pixels(depth);
        cv::Mat depthMat(rows, cols, CV_16U, (void*)buffer);

        char output[1000];
        strcpy_s(output, output_path);
        strcat_s(output, output_file_name);
        strcat_s(output, ".exr");
       

        try {
          cv::imwrite(output, depthMat);
        }
        catch (cv::Exception& e) {
          std::cout << e.msg << std::endl;
        }

        //cv::FileStorage file(output, cv::FileStorage::WRITE);

        //// Write to file!
        //file << "depthMat" << depthMat;

        ////Close the file and release all the memory buffers
        //file.release();

        k4a_image_release(depth);
        return true;
    }
    else
    {
        return false;
    }
}

bool process_mkv_offline(const char* input_path, const char* output_path, const char* output_file_name, k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT )
{
    k4a_playback_t playback_handle = nullptr;
    k4a_result_t result = k4a_playback_open(input_path, &playback_handle);
    if (result != K4A_RESULT_SUCCEEDED)
    {
        cerr << "Cannot open recording at " << input_path << endl;
        return false;
    }

    k4a_calibration_t calibration;
    result = k4a_playback_get_calibration(playback_handle, &calibration);

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

    if (result != K4A_RESULT_SUCCEEDED)
    {
        cerr << "Failed to get calibration" << endl;
        return false;
    }

    k4abt_tracker_t tracker = NULL;
    if (K4A_RESULT_SUCCEEDED != k4abt_tracker_create(&calibration, tracker_config, &tracker))
    {
        cerr << "Body tracker initialization failed!" << endl;
        return false;
    }

    json json_output;
    json_output["k4abt_sdk_version"] = K4ABT_VERSION_STR;
    json_output["source_file"] = input_path;
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
        k4a_stream_result_t stream_result = k4a_playback_get_next_capture(playback_handle, &capture_handle);
        if (stream_result == K4A_STREAM_RESULT_EOF)
        {
            break;
        }

        cout << "frame " << frame_count << '\r';
        if (stream_result == K4A_STREAM_RESULT_SUCCEEDED)
        {
            // Only try to predict joints when capture contains depth image
            if (check_depth_image_exists(capture_handle, output_path, output_file_name))
            {
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
    k4a_playback_close(playback_handle);

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

bool ProcessArguments(k4abt_tracker_configuration_t &tracker_config, int argc, char** argv)
{
    if (argc < 3)
    {
        PrintUsage();
        return false;
    }
    for( int i = 3; i < argc; i++ )
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

int main(int argc, char **argv)
{
    k4abt_tracker_configuration_t tracker_config = K4ABT_TRACKER_CONFIG_DEFAULT;
  /*  if (!ProcessArguments(tracker_config, argc, argv))
        return -1;
    return process_mkv_offline(argv[1], argv[2], tracker_config) ? 0 : -1;*/
    return process_mkv_offline("F:\\Weights_Task\\Data\\Fib_weights_original_videos\\Group_03-sub1.mkv", "F:\\Weights_Task\\Data\\", "Group_03-sub1", tracker_config) ? 0 : -1;
}
