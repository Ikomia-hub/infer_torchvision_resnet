from ikomia import core, dataprocess
from ikomia.dnn.torch import models
import os
import copy
import cv2
import torch
import torchvision.transforms as transforms


# --------------------
# - Class to handle the process parameters
# - Inherits core.CProtocolTaskParam from Ikomia API
# --------------------
class ResnetParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name_or_path = ""
        self.model_name = 'resnet18'
        self.dataset = 'ImageNet'
        self.input_size = 224
        self.model_path = ''
        self.class_file = os.path.dirname(os.path.realpath(__file__)) + "/models/imagenet_classes.txt"
        self.update = False

    def set_values(self, params):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name_or_path = params["model_name_or_path"]
        self.model_name = params["model_name"]
        self.dataset = params["dataset"]
        self.input_size = int(params["input_size"])
        self.model_path = params["model_path"]
        self.class_file = params["class_file"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        params = {
                "model_name_or_path": self.model_name_or_path,
                "model_name": self.model_name,
                "dataset": self.dataset,
                "input_size": str(self.input_size),
                "model_path": self.model_path,
                "class_file": self.class_file}
        return params


# --------------------
# - Class which implements the process
# - Inherits core.CProtocolTask or derived from Ikomia API
# --------------------
class Resnet(dataprocess.CClassificationTask):

    def __init__(self, name, param):
        dataprocess.CClassificationTask.__init__(self, name)
        self.model = None
        # Detect if we have a GPU available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Create parameters class
        if param is None:
            self.set_param_object(ResnetParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 2

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
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        # Step progress bar:
        self.emit_step_progress()

        # Load model
        if self.model is None or param.update:
            # Load class names
            self.read_class_names(param.class_file)
            # Load model
            if param.model_name_or_path != "":
                if os.path.isfile(param.model_name_or_path):
                    param.dataset = "Custom"
                    param.model_path = param.model_name_or_path
                else:
                    param.model_name = param.model_name_or_path

            use_torchvision = param.dataset != "Custom"
            self.model = models.resnet(model_name=param.model_name,
                                       use_pretrained=use_torchvision,
                                       classes=len(self.get_names()))
            if param.dataset == "Custom":
                self.model.load_state_dict(torch.load(param.model_path, map_location=self.device))

            self.model.to(self.device)
            param.update = False

        if self.is_whole_image_classification():
            image_in = self.get_input(0)
            src_image = image_in.get_image()
            predictions = self.predict(src_image, param.input_size)
            sorted_data = sorted(zip(predictions.flatten().tolist(), self.get_names()), reverse=True)
            confidences = [str(conf) for conf, _ in sorted_data]
            names = [name for _, name in sorted_data]
            self.set_whole_image_results(names, confidences)
        else:
            input_objects = self.get_input_objects()
            for obj in input_objects:
                roi_img = self.get_object_sub_image(obj)
                if roi_img is None:
                    continue

                predictions = self.predict(roi_img, param.input_size)
                class_index = predictions.argmax().item()
                self.add_object(obj, class_index, predictions[class_index].item())

        # Step progress bar:
        self.emit_step_progress()

        # Call endTaskRun to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits dataprocess.CProcessFactory from Ikomia API
# --------------------
class ResnetFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_torchvision_resnet"
        self.info.short_description = "ResNet inference model for image classification."
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
        self.info.documentation_link = "https://arxiv.org/abs/1512.03385"
        self.info.repository = "https://github.com/pytorch/vision"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Classification"
        self.info.icon_path = "icons/pytorch-logo.png"
        self.info.version = "1.2.0"
        self.info.keywords = "residual,cnn,classification"

    def create(self, param=None):
        # Create process object
        return Resnet(self.info.name, param)
