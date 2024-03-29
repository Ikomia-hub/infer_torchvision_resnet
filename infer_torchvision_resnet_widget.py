from ikomia import core, dataprocess
from ikomia.utils import qtconversion, pyqtutils
import os
from infer_torchvision_resnet.infer_torchvision_resnet_process import ResnetParam
# PyQt GUI framework
from PyQt5.QtWidgets import *


# --------------------
# - Class which implements widget associated with the process
# - Inherits core.CProtocolTaskWidget from Ikomia API
# --------------------
class ResnetWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = ResnetParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.grid_layout = QGridLayout()

        self.combo_model = pyqtutils.append_combo(self.grid_layout, "Model name")
        self.combo_model.addItem("resnet18")
        self.combo_model.addItem("resnet34")
        self.combo_model.addItem("resnet50")
        self.combo_model.addItem("resnet101")
        self.combo_model.addItem("resnet152")
        self.combo_model.setCurrentIndex(self._get_model_name_index())

        self.combo_dataset = pyqtutils.append_combo(self.grid_layout, "Trained on")
        self.combo_dataset.addItem("ImageNet")
        self.combo_dataset.addItem("Custom")
        self.combo_dataset.setCurrentIndex(self._get_dataset_index())
        self.combo_dataset.currentIndexChanged.connect(self.on_combo_dataset_changed)

        self.spin_size = pyqtutils.append_spin(self.grid_layout, label="Input size", value=self.parameters.input_size)

        self.browse_model = pyqtutils.append_browse_file(self.grid_layout, "Model path", self.parameters.model_weight_file)

        self.browse_classes = pyqtutils.append_browse_file(self.grid_layout, "Classes path", self.parameters.class_file)

        if self.parameters.dataset == "ImageNet":
            self.browse_model.set_path("Not used")
            self.browse_model.setEnabled(False)
            self.browse_classes.setEnabled(False)

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.grid_layout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def _get_model_name_index(self):
        if self.parameters.model_name == "resnet18":
            return 0
        elif self.parameters.model_name == "resnet34":
            return 1
        elif self.parameters.model_name == "resnet50":
            return 2
        elif self.parameters.model_name == "resnet101":
            return 3
        elif self.parameters.model_name == "resnet152":
            return 4
        else:
            return 0

    def _get_dataset_index(self):
        if self.parameters.dataset == "ImageNet":
            return 0
        else:
            return 1

    def on_combo_dataset_changed(self, index):
        if self.combo_dataset.itemText(index) == "ImageNet":
            self.browse_model.set_path("Not used")
            self.browse_model.setEnabled(False)
            self.browse_classes.set_path(os.path.dirname(os.path.realpath(__file__)) + "/models/imagenet_classes.txt")
            self.browse_classes.setEnabled(False)
        else:
            self.browse_model.clear()
            self.browse_model.setEnabled(True)
            self.browse_classes.clear()
            self.browse_classes.setEnabled(True)

    def on_apply(self):
        # Apply button clicked slot
        # Get parameters from widget
        self.parameters.update = True
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.dataset = self.combo_dataset.currentText()
        self.parameters.input_size = self.spin_size.value()
        self.parameters.model_weight_file = self.browse_model.path
        self.parameters.class_file = self.browse_classes.path

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits dataprocess.CWidgetFactory from Ikomia API
# --------------------
class ResnetWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_torchvision_resnet"

    def create(self, param):
        # Create widget object
        return ResnetWidget(param, None)
