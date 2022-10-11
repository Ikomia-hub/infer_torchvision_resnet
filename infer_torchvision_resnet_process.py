from ikomia import core, dataprocess
from ikomia.dnn.torch import models
import os
import copy
import cv2
import torch
import torchvision.transforms as transforms
import random


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class ResnetParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = 'resnet18'
        self.dataset = 'ImageNet'
        self.input_size = 224
        self.model_path = ''
        self.classes_path = os.path.dirname(os.path.realpath(__file__)) + "/models/imagenet_classes.txt"
        self.update = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.dataset = param_map["dataset"]
        self.input_size = int(param_map["input_size"])
        self.model_path = param_map["model_path"]
        self.classes_path = param_map["classes_path"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["model_name"] = self.model_name
        param_map["dataset"] = self.dataset
        param_map["input_size"] = str(self.input_size)
        param_map["model_path"] = self.model_path
        param_map["classes_path"] = self.classes_path
        return param_map


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class Resnet(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        self.model = None
        self.colors = None
        self.class_names = []
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Output for object classification
        self.addOutput(dataprocess.CObjectDetectionIO())
        # Outputs for whole image classification
        self.addOutput(dataprocess.CGraphicsOutput())
        self.addOutput(dataprocess.CDataStringIO())

        # Create parameters class
        if param is None:
            self.setParam(ResnetParam())
        else:
            self.setParam(copy.deepcopy(param))

    def load_class_names(self):
        self.class_names.clear()
        param = self.getParam()

        with open(param.classes_path) as f:
            for row in f:
                self.class_names.append(row[:-1])

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

    def crop_image(self, src, width, height, box):
        x = int(box[0])
        y = int(box[1])
        w = int(box[2])
        h = int(box[3])

        if x < 0:
            x = 0
        if x + w >= width:
            w = width - x - 1
        if y < 0:
            y = 0
        if y + h >= height:
            h = height - y - 1

        if w < 2 or h < 2:
            return None

        crop_img = src[y:y + h, x:x + w]
        return crop_img

    def predict(self, image, input_size):
        input_img = cv2.resize(image, (input_size, input_size))

        trs = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        input_tensor = trs(input_img).to(self.device)
        input_tensor = input_tensor.unsqueeze(0)
        prob = None

        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.nn.functional.softmax(output[0], dim=0)

        return prob

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get parameters :
        param = self.getParam()

        # Get input :
        image_in = self.getInput(0)
        src_image = image_in.getImage()
        graphics_in = self.getInput(1)

        h = src_image.shape[0]
        w = src_image.shape[1]

        # Step progress bar:
        self.emitStepProgress()

        # Load model
        if self.model is None or param.update:
            # Load class names
            self.load_class_names()
            # Load model
            use_torchvision = param.dataset != "Custom"
            self.model = models.resnet(model_name=param.model_name,
                                       use_pretrained=use_torchvision,
                                       classes=len(self.class_names))
            if param.dataset == "Custom":
                self.model.load_state_dict(torch.load(param.model_path, map_location=self.device))

            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.class_names]
            self.model.to(self.device)
            param.update = False

        objects_to_classify = []
        if graphics_in.isDataAvailable():
            for item in graphics_in.getItems():
                if not item.isTextItem():
                    objects_to_classify.append(item)

        if len(objects_to_classify) > 0:
            obj_detect_io = self.getOutput(1)

            for obj in objects_to_classify:
                # Inference
                rc = obj.getBoundingRect()
                crop_img = self.crop_image(src_image, w, h, rc)

                if crop_img is None:
                    continue

                pred = self.predict(crop_img, param.input_size)
                class_index = pred.argmax()
                obj_detect_io.addObject(obj.getId(), self.class_names[class_index], pred[class_index].item(),
                                        rc[0], rc[1], rc[2], rc[3], self.colors[class_index])
        else:
            graphics_output = self.getOutput(2)
            graphics_output.setNewLayer("ResNet")
            graphics_output.setImageIndex(0)
            table_output = self.getOutput(3)
            table_output.setOutputType(dataprocess.NumericOutputType.TABLE)
            table_output.clearData()

            pred = self.predict(src_image, param.input_size)
            # Set graphics output
            class_index = pred.argmax()
            msg = self.class_names[class_index] + ": {:.3f}".format(pred[class_index])
            graphics_output.addText(msg, 0.05 * w, 0.05 * h)
            # Set numeric output
            sorted_data = sorted(zip(pred.flatten().tolist(), self.class_names), reverse=True)
            confidences = [str(conf) for conf, _ in sorted_data]
            names = [name for _, name in sorted_data]
            table_output.addValueList(confidences, "Confidence", names)

        # Step progress bar:
        self.emitStepProgress()

        # Forward input image
        self.forwardInputImage(0, 0)

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class ResnetFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_torchvision_resnet"
        self.info.shortDescription = "ResNet inference model for image classification."
        self.info.description = "ResNet inference model for image classification. " \
                                "Implementation from PyTorch torchvision package. " \
                                "This Ikomia plugin can make inference of pre-trained model from " \
                                "ImageNet dataset or custom trained model. Custom training can be made with " \
                                "the associated MaskRCNNTrain plugin from Ikomia marketplace. Different versions " \
                                "are available with 18, 34, 50, 101 or 152 layers."
        self.info.authors = "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun"
        self.info.article = "Deep Residual Learning for Image Recognition"
        self.info.journal = "Conference on Computer Vision and Pattern Recognition (CVPR)"
        self.info.year = 2016
        self.info.licence = "BSD-3-Clause License"
        self.info.documentationLink = "https://arxiv.org/abs/1512.03385"
        self.info.repository = "https://github.com/pytorch/vision"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Classification"
        self.info.iconPath = "icons/pytorch-logo.png"
        self.info.version = "1.2.0"
        self.info.keywords = "residual,cnn,classification"

    def create(self, param=None):
        # Create process object
        return Resnet(self.info.name, param)
