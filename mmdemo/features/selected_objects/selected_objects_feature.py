from typing import final

import numpy as np

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import (
    ConesInterface,
    ObjectInterface3D,
    SelectedObjectsInterface,
)
from mmdemo.interfaces.data import Cone


@final
class SelectedObjects(BaseFeature[SelectedObjectsInterface]):
    """
    Determine which objects are selected by checking if their
    centers are contained within cones.

    Input interfaces are `ObjectInterface3D` and any number of `ConesInterface`

    Output interface is `SelectedObjectsInterface`.
    """

    def get_output(self, obj: ObjectInterface3D, *cones_list: ConesInterface):
        if not obj.is_new():
            return None

        selected_indices = set()

        for cones in cones_list:
            if not cones.is_new():
                continue

            for cone in cones.cones:
                for i in range(len(obj.objects)):
                    if self.cone_contains_point(cone, obj.objects[i].center):
                        selected_indices.add(i)

        selected = list(
            zip(obj.objects, [i in selected_indices for i in range(len(obj.objects))])
        )
        return SelectedObjectsInterface(objects=selected)

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
