from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class ResNet(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def getProcessFactory(self):
        # Instantiate process object
        from ResNet.ResNet_process import ResNetProcessFactory
        return ResNetProcessFactory()

    def getWidgetFactory(self):
        # Instantiate associated widget object
        from ResNet.ResNet_widget import ResNetWidgetFactory
        return ResNetWidgetFactory()
