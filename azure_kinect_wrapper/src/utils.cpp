#include "utils.hpp"

using namespace nlohmann;

json body_frame_info(k4abt::frame frame) {
  uint32_t num_bodies = frame.get_num_bodies();
  uint64_t timestamp = frame.get_device_timestamp().count();

  json frame_result_json;
  frame_result_json["timestamp_usec"] = timestamp;
  frame_result_json["num_bodies"] = num_bodies;
  frame_result_json["bodies"] = json::array();
  for (uint32_t i = 0; i < num_bodies; i++) {
    k4abt_skeleton_t skeleton = frame.get_body_skeleton(i);

    json body_result_json;
    int body_id = frame.get_body_id(i);
    body_result_json["body_id"] = body_id;

    for (int j = 0; j < K4ABT_JOINT_COUNT; j++) {
      body_result_json["joint_positions"].push_back(
          {skeleton.joints[j].position.xyz.x, skeleton.joints[j].position.xyz.y,
           skeleton.joints[j].position.xyz.z});

      body_result_json["joint_orientations"].push_back(
          {skeleton.joints[j].orientation.wxyz.w,
           skeleton.joints[j].orientation.wxyz.x,
           skeleton.joints[j].orientation.wxyz.y,
           skeleton.joints[j].orientation.wxyz.z});
    }
    frame_result_json["bodies"].push_back(body_result_json);
  }

  return frame_result_json;
}
