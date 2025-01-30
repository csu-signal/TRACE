from typing import final

import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    ConesInterface,
    SelectedParticipantsInterface,
    BodyTrackingInterface,
    CameraCalibrationInterface
)
from mmdemo.interfaces.data import Cone, ParticipantInfo
from mmdemo.utils.joints import Joint
from mmdemo.utils.coordinates import world_3d_to_camera_3d


@final
class SelectedParticipant(BaseFeature[SelectedParticipantsInterface]):
    """
    Determine which individuals are selected by checking if their
    centers are contained within cones.

    Input interfaces are `BodyTrackingInterface`, `CameraCalibrationInterface` and any number of `ConesInterface`.

    Output interface is `SelectedParticipantsInterface`.
    """

    def __init__(
        self,
        bt: BaseFeature[BodyTrackingInterface],
        cal: BaseFeature[CameraCalibrationInterface],
        *cones: BaseFeature[ConesInterface]
    ) -> None:
        super().__init__(bt, cal, *cones)

    def get_output(self, bt: BodyTrackingInterface, cc: CameraCalibrationInterface, *cones_list: ConesInterface):
        if not bt.is_new():
            return None
        
        parts = []
        bt = self.fix_body_id(bt) #convert the body ids to 1,2,3

        for _, body in enumerate(bt.bodies):
            #get nose coordinate
            nose = self.get_joint(Joint.NOSE, body, cc)
            part = ParticipantInfo(nosePoint=nose, participantId = body["body_id"])
            parts.append(part)

        # track minimum distance from all cones
        # which select an object for sorting
        best_selected_dist = {}
        
        for cones in cones_list:
            if not cones.is_new():
                continue

            for cone in cones.cones:
                for i in range(len(parts)):
                    if self.cone_contains_point(cone, parts[i].nosePoint):
                        dist = self.get_sorting_dist(cone, parts[i].nosePoint)
                        if i not in best_selected_dist or best_selected_dist[i] > dist:
                            best_selected_dist[i] = dist


        # [(object, selected?, closest dist to a selecting cone)]
        selected = zip(
            parts,
            [i in best_selected_dist for i in range(len(parts))],
            [
                best_selected_dist[i] if i in best_selected_dist else float("inf")
                for i in range(len(parts))
            ],
        )
        selected = list(selected)
        selected.sort(key=lambda x: x[2])
        return SelectedParticipantsInterface(participants=[(i[0], i[1]) for i in selected])

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
    
    @staticmethod
    def fix_body_id(bt): 
        # sort by head position
        bt.bodies.sort(key=lambda body: body["joint_positions"][3][0])
        # change body id according to head position relative to other participants
        for id, body in enumerate(bt.bodies):
            body["body_id"] = id + 1
        
        return bt
    
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
