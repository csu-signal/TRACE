from pathlib import Path
from typing import final

import cv2
import numpy as np
import torch

from mmdemo.base_feature import BaseFeature
from mmdemo.features.objects.config import CLASSES, DEVICE, NUM_CLASSES
from mmdemo.features.objects.model import create_model
from mmdemo.interfaces import ColorImageInterface, ObjectInterface3D
from mmdemo.interfaces.data import ObjectInfo3D
from mmdemo.utils.Gamr import Block, GamrTarget

# import helpers
# from mmdemo.features.proposition.helpers import ...

# detection_threshold = 0.6
RESIZE_TO = (512, 512)


@final
class Object(BaseFeature[ObjectInterface3D]):
    """
    A feature to get and track the objects through a scene.

    Input interface is `ColorImageInterface'.

    Output inteface is `ObjectInterface3D`.
    """

    def __init__(self, *args, detection_threshold=0.6) -> None:
        super().__init__(*args)
        self.detectionThreshold = detection_threshold

    @classmethod
    def get_input_interfaces(cls):
        return [ColorImageInterface]

    @classmethod
    def get_output_interface(cls):
        return ObjectInterface3D

    def initialize(self):
        # print("Torch Device " + str(DEVICE))
        # print("Python version " + str(platform.python_version()))

        # load the best objectModel and trained weights - for object detection
        self.device = DEVICE
        self.objectModel = create_model(num_classes=NUM_CLASSES)

        model_path = (
            Path(__file__).parent / "objectDetectionModels" / "best_model-objects.pth"
        )
        checkpoint = torch.load(str(model_path), map_location=DEVICE)

        self.objectModel.load_state_dict(checkpoint["model_state_dict"], strict=False)
        self.objectModel.to(DEVICE).eval()

        # TODO implement logger?
        # self.init_logger(log_dir)
        pass

    def get_output(self, col: ColorImageInterface) -> ObjectInterface3D | None:
        if not col.is_new():
            return None

        objects = []
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
                if found.__contains__(class_name):
                    continue

                found.append(class_name)
                p1 = [box[0], box[1]]
                p2 = [box[2], box[3]]
                center = [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]
                des = self.getDescription(float(class_name))

                if des != GamrTarget.SCALE:
                    objects.append(
                        ObjectInfo3D(p1=p1, p2=p2, center=center, object_class=des)
                    )

                    # TODO logging
                    # self.log_block(frameIndex, block)
                    # blockDescriptions.append(block.description)

        return ObjectInterface3D(objects=objects)

    def getDescription(self, classId):
        """
        `self` -- instance of object feature class
        `classId` -- the class id to be interpreted

        Returns description of the object for the class id
        """
        if classId == 0:
            return GamrTarget.RED_BLOCK

        if classId == 1:
            return GamrTarget.YELLOW_BLOCK

        if classId == 2:
            return GamrTarget.GREEN_BLOCK

        if classId == 3:
            return GamrTarget.BLUE_BLOCK

        if classId == 4:
            return GamrTarget.PURPLE_BLOCK

        if classId == 5:
            return GamrTarget.SCALE

    # def init_logger(self, log_dir):
    #     if log_dir is not None:
    #         self.logger = Logger(file=log_dir / self.LOG_FILE)
    #     else:
    #         self.logger = Logger()

    #     self.logger.write_csv_headers("frame_index", "class", "p10", "p11", "p20", "p21")

    # def log_block(self, frame_index, block: Block):
    #     self.logger.append_csv(
    #             frame_index,
    #             block.description.value,
    #             block.p1[0],
    #             block.p1[1],
    #             block.p2[0],
    #             block.p2[1]
    #     )
