from Utils.ConfigReaders.yaml_reader import YamlConfigReader
from collections import deque
import os


class AllDeque(object):
    configs = YamlConfigReader(
        os.path.join(os.path.abspath(os.path.dirname(__file__)), "../Data/configs/global.yaml")).configs_info()
    # 通信队列
    Acq2Gui_src__deque = deque([], maxlen=configs["Acq2Gui_src__DEQUE_LEN"])
    Acq2Inference_net__deque = deque([], maxlen=configs["Acq2Inference_net__DEQUE_LEN"])
    Inference2Gui__deque = deque([], maxlen=configs["Inference2Gui__DEQUE_LEN"])


if __name__ == '__main__':
    AllDeque.Acq2Gui_src__deque.append("xsf")
    print(AllDeque.Acq2Gui_src__deque.popleft())
