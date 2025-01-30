from typing import final

import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    GazeConesInterface,
    GazeSelectionInterface,
    BodyTrackingInterface,
    CameraCalibrationInterface
)
from mmdemo.interfaces.data import Cone, ParticipantInfo
from mmdemo.utils.joints import Joint
from mmdemo.utils.coordinates import world_3d_to_camera_3d


@final
class GazeSelection(BaseFeature[GazeSelectionInterface]):
    """
    Determine which individuals are selected by checking if their
    centers are contained within cones.

    Input interfaces are `BodyTrackingInterface`, `CameraCalibrationInterface` and `GazeConesInterface`.

    Output interface is `GazeSelectionInterface`.
    """

    def __init__(
        self,
        bt: BaseFeature[BodyTrackingInterface],
        cal: BaseFeature[CameraCalibrationInterface],
        gz: BaseFeature[GazeConesInterface],
        #added to unify participant ids
        left_position = -400,
        middle_position = 400
    ) -> None:
        super().__init__(bt, cal, gz)
        self.left_position = left_position
        self.middle_position = middle_position
        self.count = 0

    def get_output(self, bt: BodyTrackingInterface, cc: CameraCalibrationInterface, gz: GazeConesInterface):
        self.count += 1
        if not bt.is_new() or not gz.is_new():
            return None

        #a list containing information of different participants: the coordinate of nose and participant id
        parts = []

        for _, body in enumerate(bt.bodies):
            #get nose coordinate
            nose = self.get_joint(Joint.NOSE, body, cc)
            #unify participant ids
            x = body["joint_positions"][1][0]
            if x < self.left_position:
                body_id = "P1"
            elif x > self.left_position and x < self.middle_position:
                body_id = "P2"
            else:
                body_id = "P3"
            part = ParticipantInfo(nosePoint = nose, participantId = body_id)
            parts.append(part)

        gaze_selection = []
        selected_dist = {}

        for body, cone in zip(gz.body_ids, gz.cones):
            for i in range(len(parts)):
                if self.cone_contains_point(cone, parts[i].nosePoint) and body != parts[i].participantId:
                    dist = self.get_sorting_dist(cone, parts[i].nosePoint)
                    selected_dist[parts[i].participantId] = dist
            if selected_dist == {}:
                gaze_selection.append((body, "other"))
            else:
                gaze_selection.append((body, sorted(selected_dist.items(), key = lambda x:x[1])[0][0]))
            selected_dist = {}

        return GazeSelectionInterface(selection = gaze_selection)

    @staticmethod
    def cone_contains_point(cone: Cone, point_3d):
        # shift so base is at the origin
        point_3d = np.array(point_3d) - cone.base
        vertex = cone.vertex - cone.base

        # unit vector in direction of cone
        dir = vertex / np.linalg.norm(vertex)

        # magnitude of component parallel to cone dir
        ll = np.dot(point_3d, dir)
        max_ll = np.dot(vertex, dir)

        if ll < 0 or ll > max_ll:
            return False

        # magnitude of component perpendicular to cone dir
        # (point3d = parallel vector + perpendicular vector)
        perp = np.linalg.norm(point_3d - ll * dir)

        # maximum perpendicular component is the
        # linear interpolation between base_radius
        # and vertex radius
        max_perp = (
            cone.base_radius + (cone.vertex_radius - cone.base_radius) * ll / max_ll
        )

        if perp > max_perp:
            return False

        return True

    @staticmethod
    def get_sorting_dist(cone: Cone, point_3d):
        """
        Get the distance used for sorting objects. If two objects
        are selected by a single cone, their order will be sorted by
        this value in increasing order.
        """
        # TODO: we used to sort by perpendicular distance from projection, then by
        # distance to the projection. Both of these could have weird cases but I'm
        # not sure what the best distance to sort by is. I think absolute distance
        # might make the most sense, but change if needed. If this is changed,
        # also change the corresponding test.
        return np.linalg.norm(cone.base - np.array(point_3d))
    
    @final
    def get_joint(self, joint, body, cc):
        """
        `joint` -- the joint to be retrieved
        `body` -- the body whose joint is being retrieved
        `cc` -- instance of `CameraCalibrationInterface`

        Returns camera coordinates of requested joint
        """
        r_w = np.array(body["joint_positions"][joint.value])
        return world_3d_to_camera_3d(r_w, cc)