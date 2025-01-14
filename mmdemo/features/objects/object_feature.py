from pathlib import Path
from typing import final

import cv2
import numpy as np
import torch

from mmdemo.base_feature import BaseFeature
from mmdemo.features.objects.config import CLASSES, DEVICE, NUM_CLASSES
from mmdemo.features.objects.model import create_model
from mmdemo.interfaces import (
    CameraCalibrationInterface,
    ColorImageInterface,
    DepthImageInterface,
    ObjectInterface3D,
)
from mmdemo.interfaces.data import GamrTarget, ObjectInfo3D
from mmdemo.utils.coordinates import CoordinateConversionError, pixel_to_camera_3d

# import helpers
# from mmdemo.features.proposition.helpers import ...

# detection_threshold = 0.6
RESIZE_TO = (512, 512)


@final
class Object(BaseFeature[ObjectInterface3D]):
    """
    A feature to get and track the objects through a scene.

    Input interfaces are `ColorImageInterface, DepthImageInterface, CameraCalibrationInterface'.

    Output interface is `ObjectInterface3D`.

    Keyword arguments:
    `detection_threshold` -- confidence threshold for the object detector model
    `model_path` -- the path to the model (or None to use the default)
    """

    DEFAULT_MODEL_PATH = (
        Path(__file__).parent / "objectDetectionModels" / "best_model-objects.pth"
    )

    def __init__(
        self,
        color: BaseFeature[ColorImageInterface],
        depth: BaseFeature[DepthImageInterface],
        calibration: BaseFeature[CameraCalibrationInterface],
        *,
        detection_threshold=0.6,
        model_path: Path | None = None
    ) -> None:
        super().__init__(color, depth, calibration)
        self.detectionThreshold = detection_threshold
        if model_path is None:
            self.model_path = self.DEFAULT_MODEL_PATH
        else:
            self.model_path = model_path

    def initialize(self):
        # print("Torch Device " + str(DEVICE))
        # print("Python version " + str(platform.python_version()))

        # load the best objectModel and trained weights - for object detection
        self.device = DEVICE
        self.objectModel = create_model(num_classes=NUM_CLASSES)

        checkpoint = torch.load(str(self.model_path), map_location=DEVICE, weights_only=True)

        self.objectModel.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.objectModel.to(DEVICE).eval()

    def get_output(
        self,
        col: ColorImageInterface,
        dep: DepthImageInterface,
        calibration: CameraCalibrationInterface,
    ) -> ObjectInterface3D | None:
        if not col.is_new() or not dep.is_new():
            return None

        objects = []

        # TODO: WARNING -- the next line does not do anything
        # because `image` is redefined on the line after.
        # This is how it has been the whole time and the model
        # performs fine, but it is also slower than it would be
        # if we resized. Actually resizing seems to hurt performance
        # so it will just be left like this for now.
        image = cv2.resize(col.frame, RESIZE_TO)
        image = col.frame.astype(np.float32)

        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).to(DEVICE)
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            # get predictions for the current frame
            outputs = self.objectModel(image.to(self.device))

        # load all detection to CPU for further operations
        outputs = [{k: v.to("cpu") for k, v in t.items()} for t in outputs]
        found = []
        if len(outputs[0]["boxes"]) != 0:
            boxes = outputs[0]["boxes"].data.numpy()
            scores = outputs[0]["scores"].data.numpy()
            boxes = boxes[scores >= self.detectionThreshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]["labels"].cpu().numpy()]

            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                if class_name in found:
                    continue

                try:
                    p1 = [box[0], box[1]]
                    p2 = [box[2], box[3]]
                    center = [(p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2]
                    center3d = pixel_to_camera_3d(center, dep, calibration)
                    des = self.getDescription(float(class_name))

                    found.append(class_name)
                    if des != GamrTarget.SCALE:
                        objects.append(
                            ObjectInfo3D(
                                p1=p1, p2=p2, center=center3d, object_class=des
                            )
                        )
                except CoordinateConversionError:
                    pass

        return ObjectInterface3D(objects=objects)

    def getDescription(self, classId):
        """
        `self` -- instance of object feature class
        `classId` -- the class id to be interpreted

        Returns description of the object for the class id
        """
        if classId == 0:
            return GamrTarget.RED_BLOCK

        elif classId == 1:
            return GamrTarget.YELLOW_BLOCK

        elif classId == 2:
            return GamrTarget.GREEN_BLOCK

        elif classId == 3:
            return GamrTarget.BLUE_BLOCK

        elif classId == 4:
            return GamrTarget.PURPLE_BLOCK

        elif classId == 5:
            return GamrTarget.SCALE

        raise ValueError("Unknown class id")
