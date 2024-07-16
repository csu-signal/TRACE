from featureModules.IFeature import *
import torch
import platform
from logger import Logger
from utils import *
from featureModules.objects.model import *
from featureModules.objects.config import *

# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.6
RESIZE_TO = (512, 512)

class ObjectFeature(IFeature):
    def __init__(self, csv_log_file=None):
        print("Torch Device " + str(DEVICE))
        print("Python version " + str(platform.python_version()))
        # load the best objectModel and trained weights - for object detection
        self.objectModel = create_model(num_classes=NUM_CLASSES)
        checkpoint = torch.load('.\\featureModules\\objects\\objectDetectionModels\\best_model-objects.pth', map_location=DEVICE)
        self.objectModel.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.objectModel.to(DEVICE).eval()

        self.logger = Logger(file=csv_log_file)
        self.logger.write_csv_headers("frame_index", "objects")

    def processFrame(self, framergb, frameIndex, csvPath):
        blocks = []
        blockDescriptions = []
        image = cv2.resize(framergb, RESIZE_TO)
        image = framergb.astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        with torch.no_grad():
            # get predictions for the current frame
            outputs = self.objectModel(image.to(DEVICE))
        
        # object rendering
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]  
        found = []
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                if(found.__contains__(class_name)):
                    continue

                found.append(class_name)
                p1 = [box[0], box[1]]
                p2 = [box[2], box[3]]
                
                block = Block(float(class_name), p1, p2)
                blocks.append(block)
                blockDescriptions.append(block.description)
                # print("Found Block: " + str(block.description))
                # print(str(p1))
                # print(str(p2))
        self.logger.append_csv(frameIndex, blockDescriptions)
        return blocks
