from typing import final
import cv2 as cv

from mmdemo.base_feature import BaseFeature
from mmdemo.base_interface import BaseInterface
from mmdemo.interfaces import (  # FrameCountInterface,; GazeInterface,
    CameraCalibrationInterface,
    ColorImageInterface,
    CommonGroundInterface,
    GazeConesInterface,
    GestureConesInterface,
    SelectedObjectsInterface
)
from mmdemo.utils.coordinates import _convert2D

# import helpers
# from mmdemo.features.proposition.helpers import ...


@final
class OutputFrames(BaseFeature[ColorImageInterface]):
    @classmethod
    def get_input_interfaces(cls):
        return [
            ColorImageInterface,
            GazeConesInterface,
            GestureConesInterface,
            SelectedObjectsInterface,
            CommonGroundInterface
        ]

    @classmethod
    def get_output_interface(cls):
        return ColorImageInterface

    def initialize(self):
        # initialize prop model
        pass

    def get_output(
        self,
        color: ColorImageInterface,
        gaze: GazeConesInterface,
        gesture: GestureConesInterface,
        objects: SelectedObjectsInterface,
        common: CommonGroundInterface,
        calibration: CameraCalibrationInterface
    ):
        if not color.is_new() or not gaze.is_new() or not gesture.is_new() or not objects.is_new() or not common.is_new() or not calibration.is_new():
            return None
        
        #render gaze vectors
        for cone in gaze.cones:
            self.projectVectorLines(cone, color.frame, calibration, False, False, True)

        #render gesture vectors
        for cone in gesture.cones:
            self.projectVectorLines(cone, color.frame, calibration, True, False, False)

        #TODO render objects

        #TODO render common ground

        return color

    def projectVectorLines(self, cone, frame, calibration, includeY, includeZ, gaze):
        baseUpY, baseDownY, baseUpZ, baseDownZ = cone.conePointsBase()
        vertexUpY, vertexDownY, vertexUpZ, vertexDownZ = cone.conePointsVertex()

        if(gaze):
            yColor = (255, 107, 170)
            ZColor = (107, 255, 138)
        else:
            yColor = (255, 255, 0)
            ZColor = (243, 82, 121)

        if includeY:
            baseUp2DY = _convert2D(baseUpY, calibration.cameraMatrix, calibration.distortion)       
            baseDown2DY = _convert2D(baseDownY, calibration.cameraMatrix, calibration.distortion)    
            vertexUp2DY = _convert2D(vertexUpY, calibration.cameraMatrix, calibration.distortion)  
            vertexDown2DY = _convert2D(vertexDownY, calibration.cameraMatrix, calibration.distortion)
            
            pointUpY = (int(baseUp2DY[0]),int(baseUp2DY[1]))
            pointDownY = (int(baseDown2DY[0]),int(baseDown2DY[1]))

            vertexPointUpY = (int(vertexUp2DY[0]),int(vertexUp2DY[1]))
            vertexPointDownY = (int(vertexDown2DY[0]),int(vertexDown2DY[1]))
            
            cv.line(frame, vertexPointUpY, pointUpY, color=yColor, thickness=5)
            cv.line(frame, vertexPointDownY, pointDownY, color=yColor, thickness=5)

        if includeZ:
            vertexUp2DZ = _convert2D(vertexUpZ, calibration.cameraMatrix, calibration.distortion)
            vertexDown2DZ = _convert2D(vertexDownZ, calibration.cameraMatrix, calibration.distortion)
            baseUp2DZ = _convert2D(baseUpZ, calibration.cameraMatrix, calibration.distortion)      
            baseDown2DZ = _convert2D(baseDownZ, calibration.cameraMatrix, calibration.distortion)

            pointUpZ = (int(baseUp2DZ[0]),int(baseUp2DZ[1]))
            pointDownZ = (int(baseDown2DZ[0]),int(baseDown2DZ[1]))

            vertexPointUpZ = (int(vertexUp2DZ[0]),int(vertexUp2DZ[1]))
            vertexPpointDownZ = (int(vertexDown2DZ[0]),int(vertexDown2DZ[1]))

            cv.line(frame, vertexPointUpZ, pointUpZ, color=ZColor, thickness=5)
            cv.line(frame, vertexPpointDownZ, pointDownZ, color=ZColor, thickness=5)

