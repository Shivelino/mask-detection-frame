import os
import time
from threading import Thread
from GUI.ui_thread import UiMainThread
from Acquisition.camera import CamSingleHK, CamSingleUSB
from Acquisition.video import Video
from Utils.global_configs import AllDeque
from Utils.ConfigReaders.yaml_reader import YamlConfigReader
from Inferences.infer import NetProcess


class BaseThread(Thread):
    """线程基础类"""

    def __init__(self, *args, **kwargs):
        Thread.__init__(self, target=self.target, args=args, kwargs=kwargs)

    def target(self, *args, **kwargs):
        """线程函数"""
        pass

    def exit(self, *args, **kwargs):
        """退出函数。Thread不自带线程退出，新加一个退出函数用于线程退出"""
        pass


class AcquisitionThread(BaseThread):
    """图像获取线程"""

    def __init__(self):
        BaseThread.__init__(self)
        self.configs = YamlConfigReader(
            os.path.join(os.path.abspath(os.path.dirname(__file__)), "../Data/configs/acquisition.yaml")).configs_info()
        if self.configs["mode"] == "hk":
            self.acq = CamSingleHK(**self.configs)
        if self.configs["mode"] == "usb":
            self.acq = CamSingleUSB(**self.configs)
        elif self.configs["mode"] == "video":
            self.acq = Video(**self.configs)
        else:
            raise Exception("Unsupported mode.")  # TODO: 之后考虑用啥错误

    def target(self):
        self.acq.run({"acq2gui_src": AllDeque.Acq2Gui_src__deque,
                      "acq2inference_net": AllDeque.Acq2Inference_net__deque
                      })

    def exit(self, *args, **kwargs):
        pass


class ProcessThread(BaseThread):
    """处理线程。神经网络等"""

    def __init__(self):
        BaseThread.__init__(self)
        self.process = NetProcess()

    def target(self, *args, **kwargs):
        self.process.run({"acq2inference_net": AllDeque.Acq2Inference_net__deque,
                          "inference2gui": AllDeque.Inference2Gui__deque
                          })


class GuiThread(BaseThread):
    """GUI线程"""

    def __init__(self):
        BaseThread.__init__(self)
        self.ui_main_thread = UiMainThread()

    def target(self):
        self.ui_main_thread.run({"acq2gui_src": AllDeque.Acq2Gui_src__deque,
                                 "inference2gui": AllDeque.Inference2Gui__deque
                                 })


if __name__ == "__main__":
    ta = AcquisitionThread()
    tp = ProcessThread()
    tg = GuiThread()  # TODO: 在开启gui线程的时候另外两个线程会出现短暂降速的现象
    ta.start()
    time.sleep(0.1)
    tp.start()
    time.sleep(0.1)
    tg.start()
