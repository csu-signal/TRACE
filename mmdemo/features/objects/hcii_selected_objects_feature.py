from typing import final

import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.features.gesture.hcii_gesture_feature import HciiGesture
from mmdemo.interfaces import (
    ConesInterface,
    HciiGestureConesInterface,
    ObjectInterface3D,
    SelectedObjectsInterface,
)
from mmdemo.interfaces.data import Cone


@final
class HciiSelectedObjects(BaseFeature[SelectedObjectsInterface]):
    """
    Determine which objects are selected by checking if their
    centers are contained within cones.

    Input interfaces are `ObjectInterface3D` and any number of `HciiGestureConesInterface`

    Output interface is `SelectedObjectsInterface`.
    """

    def __init__(
        self,
        objects: BaseFeature[ObjectInterface3D],
        gesture: BaseFeature[HciiGesture]
    ) -> None:
        super().__init__(objects, gesture)

    def get_output(self, obj: ObjectInterface3D, gesture: HciiGestureConesInterface):
        if not obj.is_new():
            return None

        # track minimum distance from all cones
        # which select an object for sorting
        best_selected_dist = {}
        
        
        for index, cone in enumerate(gesture.cones):
            if not gesture.is_new():
                continue

            #for cone in cones:
            for i in range(len(obj.objects)):
                if self.cone_contains_point(cone, obj.objects[i].center):
                    dist = self.get_sorting_dist(cone, obj.objects[i].center)
                    if i not in best_selected_dist or best_selected_dist[i] > dist:
                        best_selected_dist[i] = dist
                    obj.objects[i].wtd_id.append(gesture.wtd_body_ids[index])


        # [(object, selected?, closest dist to a selecting cone)]
        selected = zip(
            obj.objects,
            [i in best_selected_dist for i in range(len(obj.objects))],
            [
                best_selected_dist[i] if i in best_selected_dist else float("inf")
                for i in range(len(obj.objects))
            ],
        )

        selected = list(selected)
        selected.sort(key=lambda x: x[2])

        return SelectedObjectsInterface(objects=[(i[0], i[1]) for index, i in enumerate(selected)])

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
