from ikomia import dataprocess


# --------------------
# - Interface class to integrate the process with Ikomia application
# - Inherits dataprocess.CPluginProcessInterface from Ikomia API
# --------------------
class IkomiaPlugin(dataprocess.CPluginProcessInterface):

    def __init__(self):
        dataprocess.CPluginProcessInterface.__init__(self)

    def get_process_factory(self):
        # Instantiate process object
        from infer_torchvision_resnet.infer_torchvision_resnet_process import ResnetFactory
        return ResnetFactory()

    def get_widget_factory(self):
        # Instantiate associated widget object
        from infer_torchvision_resnet.infer_torchvision_resnet_widget import ResnetWidgetFactory
        return ResnetWidgetFactory()
