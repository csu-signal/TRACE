from typing import final

from mmdemo.base_feature import BaseFeature
from mmdemo.interfaces import ColorImageInterface, ObjectInterface

# import helpers
# from mmdemo.features.proposition.helpers import ...

# detection_threshold = 0.6
# RESIZE_TO = (512, 512)


@final
class Object(BaseFeature):
    @classmethod
    def get_input_interfaces(cls):
        return [ColorImageInterface]

    @classmethod
    def get_output_interface(cls):
        return ObjectInterface

    def initialize(self):
        # print("Torch Device " + str(DEVICE))
        # print("Python version " + str(platform.python_version()))
        # # load the best objectModel and trained weights - for object detection
        # self.objectModel = create_model(num_classes=NUM_CLASSES)

        # model_path = Path(__file__).parent / "objectDetectionModels" / "best_model-objects.pth"
        # checkpoint = torch.load(str(model_path), map_location=DEVICE)

        # self.objectModel.load_state_dict(checkpoint['model_state_dict'], strict=False)
        # self.objectModel.to(DEVICE).eval()

        # self.init_logger(log_dir)
        pass

    def get_output(self, col: ColorImageInterface):
        if not col.is_new():
            return None

        # call move classifier, create interface, and return

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

    # def processFrame(self, framergb, frameIndex):
    #     blocks = []
    #     blockDescriptions = []
    #     image = cv2.resize(framergb, RESIZE_TO)
    #     image = framergb.astype(np.float32)
    #     # make the pixel range between 0 and 1
    #     image /= 255.0
    #     # bring color channels to front
    #     image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    #     # convert to tensor
    #     image = torch.tensor(image, dtype=torch.float).cuda()
    #     # add batch dimension
    #     image = torch.unsqueeze(image, 0)
    #     with torch.no_grad():
    #         # get predictions for the current frame
    #         outputs = self.objectModel(image.to(DEVICE))

    #     # object rendering
    #     # load all detection to CPU for further operations
    #     outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    #     found = []
    #     if len(outputs[0]['boxes']) != 0:
    #         boxes = outputs[0]['boxes'].data.numpy()
    #         scores = outputs[0]['scores'].data.numpy()
    #         boxes = boxes[scores >= detection_threshold].astype(np.int32)
    #         draw_boxes = boxes.copy()
    #         # get all the predicited class names
    #         pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

    #         # draw the bounding boxes and write the class name on top of it
    #         for j, box in enumerate(draw_boxes):
    #             class_name = pred_classes[j]
    #             if(found.__contains__(class_name)):
    #                 continue

    #             found.append(class_name)
    #             p1 = [box[0], box[1]]
    #             p2 = [box[2], box[3]]

    #             block = Block(float(class_name), p1, p2)

    #             if(block.description != GamrTarget.SCALE):
    #                 blocks.append(block)
    #                 self.log_block(frameIndex, block)

    #                 blockDescriptions.append(block.description)
    #             # print("Found Block: " + str(block.description))
    #             # print(str(p1))
    #             # print(str(p2))
    #     return blocks
