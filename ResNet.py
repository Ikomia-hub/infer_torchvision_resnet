from ikomia import dataprocess
import ResNet_process as processMod
import ResNet_widget as widgetMod


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class ResNet(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        return processMod.ResNetProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        return widgetMod.ResNetWidgetFactory()
